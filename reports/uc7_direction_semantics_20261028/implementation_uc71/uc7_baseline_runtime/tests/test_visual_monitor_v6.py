"""V6 attachment counterexamples and explicit persistent correspondence reuse."""
import copy
import numpy as np
import pytest

from gpt_field.visual_monitor_v6 import WindowedAttachmentEvidence, RoleAwareVisualMonitorV6
from test_visual_monitor_v5 import textured, rgb, transform


def update(evidence, payload=(0., 1.), tool=(0., 1.), anchor=(0., 0.), **kwargs):
    evidence.observe(payload, tool, anchor, measured_distance_m=kwargs.pop("distance", .002),
                     separation_px=30., **kwargs)


def supported():
    evidence = WindowedAttachmentEvidence()
    for _ in range(5):
        update(evidence)
    assert evidence.observed
    return evidence


def test_one_small_noisy_direction_change_does_not_erase_coherent_lift():
    evidence = supported()
    # Observed trial002 action200: both regions moved about .65px; per-frame
    # cosine=.7855, while the preceding lift established consistent co-motion.
    for _ in range(4):
        update(evidence, (0., -.8), (0., -.8))
    update(evidence, (-.1530356, -.6219987), (.2734251, -.6025213))
    assert evidence.observed
    assert evidence.control_limit() == (None, None)


@pytest.mark.parametrize("payload,tool", [((0., 0.), (0., 1.)), ((0., 1.), (0., 0.)), ((0., -1.), (0., 1.))])
def test_static_or_oppositely_moving_feature_contradicts_attachment(payload, tool):
    evidence = supported()
    for _ in range(4):
        update(evidence, payload, tool)
    assert not evidence.observed
    assert evidence.control_limit()[0] == 0
    assert not evidence.had_observed_evidence


def test_shared_background_motion_does_not_create_attachment_evidence():
    evidence = WindowedAttachmentEvidence()
    for _ in range(8):
        update(evidence, (.7, .3), (.7, .3), (.7, .3))
    assert not evidence.observed
    assert not evidence.had_observed_evidence
    assert evidence.control_limit()[0] == 0


def test_two_static_background_seeds_cannot_validate_measured_robot_motion():
    evidence = WindowedAttachmentEvidence()
    for _ in range(8):
        update(evidence, (0., 0.), (0., 0.))
    assert not evidence.observed
    assert evidence.control_limit()[0] == 0


def test_slow_relative_slip_is_detected_by_window_instead_of_per_frame_noise():
    evidence = supported()
    for _ in range(4):
        update(evidence, (.55, 1.), (0., 1.))
    assert not evidence.observed
    assert evidence.control_limit()[0] == 0
    assert not evidence.had_observed_evidence


@pytest.mark.parametrize("kw,state", [({"opening": True}, "released"),
    ({"finger_change_m": .003}, "unobserved"), ({"reliable": False}, "correspondence_lost"),
    ({"distinct_features": False}, "contradicted"), ({"rotation_rad": .04}, "unobserved")])
def test_release_identity_and_domain_changes_immediately_clear_evidence(kw, state):
    evidence = supported()
    update(evidence, **kw)
    assert evidence.state == state
    assert not evidence.observed
    assert evidence.control_limit()[0] == 0
    assert not evidence.had_observed_evidence


def test_large_one_frame_slip_clears_even_after_long_good_history():
    evidence = supported()
    update(evidence, (4., 1.), (0., 1.))
    assert evidence.state == "contradicted"
    assert evidence.control_limit()[0] == 0


def test_stationary_hold_does_not_invent_evidence_or_erase_previous_lift():
    evidence = supported()
    n = evidence.samples
    for _ in range(7):
        update(evidence, (0., 0.), (0., 0.), distance=0.)
    assert evidence.observed
    assert evidence.samples <= n+1  # Sliding past motion may supply one window.


def test_ambiguous_after_observed_motion_has_finite_measured_travel_budget():
    evidence = supported()
    for _ in range(3):
        # A moving stationary anchor is uncertainty, not a newly validated grasp.
        update(evidence, (.1, 1.), (.1, 1.), (.6, 0.), distance=.001)
    assert evidence.state == "uncertain"
    assert not evidence.observed
    cap, reason = evidence.control_limit()
    assert 0 < cap <= .003 and reason is None
    for _ in range(5):
        update(evidence, (.1, 1.), (.1, 1.), (.6, 0.), distance=.001)
    assert evidence.control_limit() == (0., "uncertain_attachment_travel_exhausted")


def references(mode="seed", payload=False):
    return {"external": {"update": mode, "moving_id": "object" if payload else "tool",
        "moving_kind": "payload" if payload else "tool", "target_id": "anchor", "goal_offset_uv": [0., 0.],
        "seed": None if mode == "retain" else {"moving_point_uv": [.3, .4], "moving_roi_uv": [.2, .3, .4, .5],
            "target_anchor_uv": [.7, .7], "target_roi_uv": [.6, .6, .8, .8],
            "tool_point_uv": [.7, .3] if payload else None,
            "tool_roi_uv": [.6, .2, .8, .4] if payload else None}},
        "wrist": {"update": mode, "target_id": "anchor", "desired_target_uv": [.5, .5],
            "seed": None if mode == "retain" else {"target_anchor_uv": [.7, .7], "target_roi_uv": [.6, .6, .8, .8]}}}


def monitor_fixture(payload=False):
    image = rgb(textured())
    images = {"external": image, "wrist": image}
    monitor = RoleAwareVisualMonitorV6()
    state = np.zeros(8)
    monitor.configure(images, references(payload=payload), state, intent_id=1)
    return monitor, images, state


def test_explicit_retain_keeps_exact_bundle_and_goal_offset_without_model_coordinates():
    monitor, images, state = monitor_fixture()
    old = monitor.tracks.copy()
    epoch = monitor.epoch
    monitor.configure(images, references("retain"), state, intent_id=2)
    assert monitor.epoch == epoch
    assert monitor.reference_update_status == "external_retained"
    assert all(monitor.tracks[k] is v for k, v in old.items())
    assert monitor.refs["external"]["moving_point_uv"] == pytest.approx(old["moving"].uv)
    summary = monitor.summary()
    t = summary["tracking"]["moving"]
    assert t["tracked_point_uv"] == pytest.approx([.3, .4], abs=.001)
    assert t["current_roi_uv"] == pytest.approx([.2, .3, .4, .5], abs=.001)
    assert t["identity"] == "tool" and t["point_currently_observed"]


def test_seed_is_explicit_correspondence_reset_even_with_identical_coordinates():
    monitor, images, state = monitor_fixture(payload=True)
    old = monitor.tracks["moving"]
    monitor.attachment = supported(); monitor._sync_attachment()
    monitor.configure(images, references(payload=True), state, intent_id=2)
    assert monitor.tracks["moving"] is not old
    assert not monitor.attachment_evidence
    assert monitor.attachment_control_limit()[0] == 0


@pytest.mark.parametrize("failure", ["identity", "lost", "new_image", "finger", "rotation"])
def test_failed_retain_never_reseeds_or_reuses_old_attachment(failure):
    monitor, images, state = monitor_fixture(payload=True)
    monitor.attachment = supported(); monitor._sync_attachment()
    ref = references("retain", payload=True)
    if failure == "identity": ref["external"]["moving_id"] = "other_object"
    elif failure == "lost": monitor.tracks["moving"]._fail("occluded")
    elif failure == "new_image": images = {"external": rgb(textured(94)), "wrist": images["wrist"]}
    elif failure == "finger": state[6] = .01
    else: state[3] = .1
    monitor.configure(images, ref, state, intent_id=2)
    assert monitor.reference_update_status == "external_retain_rejected"
    assert monitor.refs["external"] is None
    assert not monitor._external_reliable()
    assert not monitor.attachment_evidence
    assert monitor.control_limit()[0] == 0


def test_new_intent_does_not_renew_uncertain_attachment_travel():
    monitor, images, state = monitor_fixture(payload=True)
    monitor.attachment = supported()
    monitor.attachment._uncertain(.004, "test_ambiguity")
    monitor._sync_attachment()
    monitor.configure(images, references("retain", payload=True), state, intent_id=2)
    assert not monitor.attachment_evidence
    assert monitor.attachment_control_limit() == pytest.approx((.001, None))


def test_same_features_cannot_play_payload_and_independent_tool_roles():
    monitor, images, state = monitor_fixture(payload=True)
    assert not monitor._distinct(monitor.tracks["moving"], monitor.tracks["moving"])


def test_external_seed_does_not_erase_independent_retained_wrist_identity():
    monitor, images, state = monitor_fixture()
    wrist = monitor.tracks["wrist_target"]
    refs = references(); refs["wrist"] = references("retain")["wrist"]
    monitor.configure(images, refs, state, intent_id=2)
    assert monitor.tracks["wrist_target"] is wrist
