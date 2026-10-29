"""Recent physical response and nonrenewable local-correspondence evidence."""
import copy
import numpy as np
import pytest

from gpt_field.observation import decode_image
from gpt_field.visual_monitor_v7 import RoleAwareVisualMonitorV7
from test_visual_field_v6_policy import observation, prediction
from test_visual_monitor_v6 import references
from test_visual_monitor_v5 import textured, rgb


def views(obs):
    return {"external": np.asarray(decode_image(obs.image)), "wrist": np.asarray(decode_image(obs.wrist_image))}


def plant_monitor(payload=False):
    obs = observation(payload=payload, fingers=(.01, -.01))
    monitor = RoleAwareVisualMonitorV7()
    refs = prediction(payload=payload)["visual_references"]
    monitor.configure(views(obs), refs, obs.state, intent_id=1)
    return monitor, refs, obs


def static_monitor():
    pixels = rgb(textured())
    images = {"external": pixels, "wrist": pixels}
    state = np.zeros(8)
    monitor = RoleAwareVisualMonitorV7()
    monitor.configure(images, references(), state, intent_id=1)
    return monitor, images, state


def static_transition(monitor, images, state, count=4):
    for _ in range(count):
        state[0] += .002
        monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])


def test_first_use_allows_task_motion_without_mandatory_calibration():
    monitor, _, _ = plant_monitor()
    assert monitor.control_limit() == (.005, None)
    assert not monitor.summary()["reference_health"]["repair_required"]


def test_stationary_background_rejects_after_recent_measured_motion():
    monitor, images, state = static_monitor()
    static_transition(monitor, images, state)
    assert monitor.control_limit()[0] == 0
    health = monitor.summary()["reference_health"]
    assert health["repair_required"]
    assert health["status"] == "controlled_reference_unresponsive_change_correspondence"
    assert health["recent_window"]["feature_displacement_px"] < health["recent_window"]["motion_threshold_px"]
    assert "different visible rigid landmark" in health["recommended_action"]
    assert health["rejected_current_uv"] == pytest.approx([.3, .4], abs=.001)


def test_new_intent_and_retain_never_renew_uncertain_correspondence_allowance():
    monitor, images, state = static_monitor()
    for n in range(1, 10):
        state[0] += .001
        monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])
        used = monitor.summary()["reference_health"]["unvalidated_travel_m"]
        monitor.configure(images, references("retain"), state, intent_id=n+1)
        assert monitor.summary()["reference_health"]["unvalidated_travel_m"] == used
    assert monitor.control_limit()[0] == 0
    assert used == pytest.approx(.009)


@pytest.mark.parametrize("change", ["same_seed", "rename", "nearby_seed", "different_goal", "different_anchor"])
def test_reseeding_same_physical_texture_cannot_erase_rejection(change):
    monitor, images, state = static_monitor()
    static_transition(monitor, images, state)
    before = monitor.summary()["reference_health"]
    refs = references()
    if change == "rename": refs["external"]["moving_id"] = "new_label"
    if change == "nearby_seed": refs["external"]["seed"]["moving_point_uv"][0] += .009
    if change == "different_goal": refs["external"]["goal_offset_uv"] = [.1, .1]
    if change == "different_anchor":
        refs["external"]["target_id"] = "other_anchor"
        refs["external"]["seed"]["target_anchor_uv"] = [.72, .72]
    monitor.configure(images, refs, state, intent_id=20)
    after = monitor.summary()["reference_health"]
    assert after["correspondence_id"] == before["correspondence_id"]
    assert after["unvalidated_travel_m"] == before["unvalidated_travel_m"]
    assert after["seed_reused_rejected_footprint"]
    assert monitor.control_limit()[0] == 0


def test_materially_new_local_texture_gets_a_budget_but_alternating_rejected_seeds_does_not():
    monitor, images, state = static_monitor()
    static_transition(monitor, images, state)
    first_id = monitor.summary()["reference_health"]["correspondence_id"]
    second = references()
    second["external"]["seed"].update(moving_point_uv=[.7, .3], moving_roi_uv=[.6, .2, .8, .4])
    monitor.configure(images, second, state, intent_id=2)
    second_id = monitor.summary()["reference_health"]["correspondence_id"]
    assert second_id != first_id
    assert monitor.control_limit() == (.005, None)
    static_transition(monitor, images, state)
    assert monitor.control_limit()[0] == 0
    for n, ref, ident in [(3, references(), first_id), (4, second, second_id), (5, references(), first_id)]:
        monitor.configure(images, ref, state, intent_id=n)
        assert monitor.summary()["reference_health"]["correspondence_id"] == ident
        assert monitor.control_limit()[0] == 0


def test_different_foreground_texture_at_old_bad_location_is_not_a_false_identity_match():
    monitor, images, state = static_monitor()
    static_transition(monitor, images, state)
    old = monitor.summary()["reference_health"]["correspondence_id"]
    new = rgb(textured(981))
    monitor.configure({"external": new, "wrist": new}, references(), state, intent_id=2)
    assert monitor.summary()["reference_health"]["correspondence_id"] != old
    assert monitor.control_limit() == (.005, None)


def test_current_measured_predictive_motion_clears_debt_and_supports_longer_field_reuse():
    monitor, refs, obs = plant_monitor()
    for n in range(1, 29):
        assert monitor.control_limit()[0] != 0, monitor.summary()["reference_health"]
        obs = observation((.001*n, 0., 0.), fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
        if n == 8:
            keep = copy.deepcopy(refs); keep["external"].update(update="retain", seed=None)
            monitor.configure(views(obs), keep, obs.state, intent_id=2)
    assert monitor.response_ready
    assert monitor.summary()["reference_health"]["unvalidated_travel_m"] == 0.
    assert monitor.control_limit() == (None, None)


def test_old_large_displacement_cannot_hide_a_recent_stationary_feature():
    monitor, refs, obs = plant_monitor()
    for n in range(1, 11):
        obs = observation((.001*n, 0., 0.), fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.response_ready
    frozen = views(obs)
    for n in range(11, 21):
        state = list(obs.state); state[0] += .001*(n-10)
        monitor.observe(frozen, state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.maximum_feature_displacement_px > 8
    assert monitor.summary()["reference_health"]["recent_window"]["feature_displacement_px"] < .01
    assert monitor.control_limit()[0] == 0
    assert not monitor.response_ready


def test_observed_v6_payload_tool_lift_and_retention_are_unchanged():
    monitor, refs, obs = plant_monitor(payload=True)
    for n in range(1, 7):
        obs = observation((.001*n, 0., 0.), payload=True, fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.attachment_evidence
    assert monitor.attachment_control_limit() == (None, None)
    refs["external"].update(update="retain", seed=None)
    monitor.configure(views(obs), refs, obs.state, intent_id=2)
    assert monitor.attachment_evidence
    monitor.observe(views(obs), obs.state, [0, 0, 0, 0, 0, 0, -1])
    assert not monitor.attachment_evidence
    assert monitor.attachment.summary()["state"] == "released"


def test_seed_domain_change_does_not_renew_rejected_background_allowance():
    monitor, images, state = static_monitor()
    static_transition(monitor, images, state)
    state[3] = .1
    state[6] = .02
    monitor.configure(images, references(), state, intent_id=2)
    assert monitor.control_limit()[0] == 0


@pytest.mark.parametrize("payload", [False, True])
def test_fresh_direct_motion_can_rehabilitate_same_previously_stationary_correspondence(payload):
    monitor, refs, obs = plant_monitor(payload=payload)
    frozen = views(obs)
    start = np.array(obs.state)
    for n in range(1, 9):
        state = start.copy(); state[0] += .001*n
        monitor.observe(frozen, state, [.1, 0, 0, 0, 0, 0, 0])
    rejected = monitor.summary()["reference_health"]
    assert rejected["repair_required"]
    keep = copy.deepcopy(refs); keep["external"].update(update="retain", seed=None)
    monitor.configure(frozen, keep, state, intent_id=2)
    assert monitor.control_limit()[0] == 0
    for n in range(1, 15):
        moved = observation((.001*n, 0., 0.), payload=payload, fingers=(.01, -.01))
        state = np.array(moved.state); state[0] += .008
        monitor.observe(views(moved), state, [.1, 0, 0, 0, 0, 0, 0])
        if n <= 3:
            assert monitor.control_limit()[0] == 0, "Old data cannot immediately rehabilitate a reference"
        if monitor.response_ready:
            break
    assert monitor.response_ready
    health = monitor.summary()["reference_health"]
    assert health["correspondence_id"] == rejected["correspondence_id"]
    assert health["evidence_based_recoveries"] == 1
    assert health["fresh_recovery_supported_transitions"] >= 3
    assert health["unvalidated_travel_m"] == 0
    assert not health["repair_required"]
    if payload:
        assert monitor.attachment_evidence


def test_static_reference_never_recovers_from_post_rejection_retains_and_observations():
    monitor, images, state = static_monitor()
    static_transition(monitor, images, state)
    for n in range(20):
        static_transition(monitor, images, state, count=1)
        monitor.configure(images, references("retain"), state, intent_id=n+2)
        assert monitor.control_limit()[0] == 0
    health = monitor.summary()["reference_health"]
    assert health["evidence_based_recoveries"] == 0
    assert health["fresh_recovery_supported_transitions"] == 0


def test_archived_rgb_wrong_base_is_rejected_but_correct_housing_supports_field_reuse():
    """Conditional observer replay; no simulator, inference or task-result claim.

    After a hypothetical stop we still read archived observations to test later
    correspondences. This cannot establish a new rollout's duration or success.
    """
    import json
    from pathlib import Path

    trace = (Path(__file__).resolve().parents[1]/"runs/visual_field_v6_zero_shot_001"
             /"trials/bowl_original_zero_hybrid_v6/trace.jsonl")
    if not trace.exists():
        pytest.skip("The optional recorded RGB/state archive is not distributed with source")
    monitor = RoleAwareVisualMonitorV7()
    observation_before = None
    rows, configured = {}, {}

    def archived_views(value):
        return {"external": np.asarray(decode_image(value["observation/image"])),
                "wrist": np.asarray(decode_image(value["observation/wrist_image"]))}

    with trace.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("step", 0) > 236:
                break
            if event["event"] == "reset":
                observation_before = event["observation"]
            elif event["event"] == "action":
                value = event["diagnostics"].get("prediction")
                if value:
                    monitor.configure(archived_views(observation_before), value["visual_references"],
                        observation_before["observation/state"], intent_id=event["diagnostics"]["chunk_index"])
                    configured[event["step"]] = monitor.summary()["reference_health"]
            elif event["event"] == "executed":
                observation_before = event["observation"]
                summary = monitor.observe(archived_views(observation_before),
                    observation_before["observation/state"], event["action"])
                rows[event["step"]] = (monitor.control_limit(), summary)
    first_rejected = next(step for step, (limit, _) in rows.items() if limit[0] == 0)
    assert first_rejected < 10
    assert rows[first_rejected][1]["reference_health"]["unvalidated_travel_m"] < .01
    # Re-requesting the black base after partial-feature drift must still refer
    # to the rejected footprint; the later gray housing is a different seed.
    assert configured[162]["correspondence_id"] == configured[0]["correspondence_id"]
    assert configured[162]["repair_required"]
    assert configured[168]["correspondence_id"] != configured[162]["correspondence_id"]
    assert not configured[168]["repair_required"]
    assert all(rows[step][0][0] != 0 for step in range(168, 189))
    assert sum(rows[step][1]["response_validated"] for step in range(168, 189)) >= 15
    assert all(rows[step][0][0] != 0 for step in range(207, 236))
    assert sum(rows[step][1]["response_validated"] for step in range(207, 236)) >= 18
    assert not rows[236][1]["external_reliable"]
