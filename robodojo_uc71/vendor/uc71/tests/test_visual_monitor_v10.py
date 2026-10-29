"""Adaptive attachment positives, physical counterexamples and recorded RGB."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import decode_image
from gpt_field.visual_monitor_v10 import AdaptiveAttachmentEvidence, RoleAwareVisualMonitorV10
from gpt_field.visual_monitor_v6 import WindowedAttachmentEvidence
from test_visual_field_v6_policy import observation, prediction
from test_visual_monitor_v5 import rgb, textured
from test_visual_monitor_v6 import references


def advance(evidence, payload=(0., 1.), tool=(0., 1.), anchor=(0., 0.),
            distance=.001, **kwargs):
    evidence.observe(payload, tool, anchor, measured_distance_m=distance,
                     separation_px=30., **kwargs)


def supported():
    value = AdaptiveAttachmentEvidence()
    for _ in range(3):
        advance(value)
    assert value.observed
    return value


def views(obs):
    return {"external": np.asarray(decode_image(obs.image)),
            "wrist": np.asarray(decode_image(obs.wrist_image))}


def test_noisy_slow_rigid_motion_uses_displacement_not_four_frame_window():
    current, legacy = supported(), WindowedAttachmentEvidence()
    for _ in range(4):
        advance(legacy)
    observed_events = 0
    for step in range(90):
        # Endpoint noise oscillates; true common displacement is .10 px/frame.
        jitter = .025*((-1)**step)
        payload, tool = (jitter, .1), (-jitter, .1)
        for evidence in (current, legacy):
            advance(evidence, payload, tool, distance=.0003, noise_px=(.12, .12, .08))
        cap, reason = current.control_limit()
        assert reason is None and (cap is None or cap >= .001)
        assert current.uncertain_travel_m <= .005
        observed_events += current.observed
    assert observed_events >= 8
    assert legacy.control_limit()[0] == 0
    assert current.samples >= 9
    assert len(current.window) <= 64


@pytest.mark.parametrize("payload,tool", [((0., 0.), (0., .4)),
    ((0., .4), (0., 0.)), ((0., -.4), (0., .4)), ((.55, 1.), (0., 1.))])
def test_fresh_history_rejects_resolved_slip_after_good_motion(payload, tool):
    value = supported()
    for _ in range(8):
        advance(value, payload, tool)
        if value.state == "contradicted":
            break
    assert value.state == "contradicted"
    assert value.control_limit()[0] == 0 and not value.had_observed_evidence
    assert value.summary()["rejection_metrics"] is not None


def test_static_payload_never_acquires_attachment():
    value = AdaptiveAttachmentEvidence()
    for _ in range(16):
        advance(value, (0., 0.), (0., .25))
        assert not value.observed and value.control_limit()[0] == 0


def test_common_camera_flow_or_stationary_pair_never_acquires_attachment():
    for common in [(0., 0.), (.4, .2)]:
        value = AdaptiveAttachmentEvidence()
        for _ in range(20):
            advance(value, common, common, common)
        assert value.control_limit()[0] == 0 and not value.had_observed_evidence


@pytest.mark.parametrize("change,state", [({"opening": True}, "released"),
    ({"reliable": False}, "correspondence_lost"),
    ({"finger_change_m": .003}, "unobserved"),
    ({"rotation_rad": .04}, "unobserved"),
    ({"distinct_features": False}, "contradicted")])
def test_release_correspondence_or_domain_change_immediately_invalidates(change, state):
    value = supported()
    advance(value, **change)
    assert value.state == state and value.control_limit()[0] == 0
    assert not value.had_observed_evidence


def test_uncertainty_hands_off_at_sub_mm_remaining_without_rounding_up():
    value = supported()
    for _ in range(14):
        advance(value, (0., 0.), (0., 0.), distance=.0003)
    assert value.uncertain_travel_m == pytest.approx(.0042)
    assert value.control_limit() == (0., value.handoff_reason)
    assert value.summary()["uncertain_travel_remaining_m"] == pytest.approx(.0008)
    used, count = value.uncertain_travel_m, value.samples
    for _ in range(80):
        advance(value, (0., 0.), (0., 0.), distance=0.)
    assert value.control_limit() == (0., value.handoff_reason)
    assert value.uncertain_travel_m == used and value.samples == count
    assert len(value.window) == 64


def test_fresh_observed_window_is_not_reused_to_renew_uncertainty():
    value = supported()
    count = value.samples
    assert len(value.window) == 0
    for _ in range(12):
        advance(value, (0., 0.), (0., 0.), distance=0.)
    assert value.observed and value.samples == count
    advance(value, (0., .05), (0., .05), distance=.001)
    assert value.state == "uncertain" and value.samples == count
    assert value.uncertain_travel_m == pytest.approx(.001)


def test_allowance_uses_only_supported_response_and_is_frozen_during_uncertainty():
    value = AdaptiveAttachmentEvidence()
    for _ in range(12):
        advance(value, (0., .1), (0., .1), distance=.001)
        if value.observed:
            break
    assert value.observed
    assert .005 < value.uncertain_travel_limit_m <= .015
    limit = value.uncertain_travel_limit_m
    for _ in range(15):
        advance(value, (0., 0.), (0., 0.), distance=.001, noise_px=(2., 2., 2.))
        assert value.uncertain_travel_limit_m == limit
    assert value.control_limit()[0] == 0
    assert value.uncertain_travel_m >= limit


def test_cold_static_reference_is_inconclusive_until_same_finite_debt_exhausts():
    pixels = rgb(textured()); images = {"external": pixels, "wrist": pixels}
    state = np.zeros(8); monitor = RoleAwareVisualMonitorV10()
    monitor.configure(images, references(), state, intent_id=1)
    for n in range(1, 15):
        state[0] = .001*n
        monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])
        if n <= 12:
            assert monitor.control_limit()[0] > 0
            assert not monitor.summary()["reference_health"]["repair_required"]
        monitor.configure(images, references("retain"), state, intent_id=n+1)
    assert monitor.control_limit() == (0., monitor.uncertainty_budget_reason)
    used = monitor.summary()["reference_health"]["unvalidated_travel_m"]
    seed = references(); seed["external"]["moving_id"] = "renamed_same_physical_patch"
    monitor.configure(images, seed, state, intent_id=40)
    assert monitor.control_limit() == (0., monitor.uncertainty_budget_reason)
    assert monitor.summary()["reference_health"]["unvalidated_travel_m"] == used
    assert "inconclusive" in monitor.summary()["reference_health"]["recommended_action"]


def test_real_tracker_slow_rigid_payload_keeps_a_useful_allowance_without_reseeding():
    obs = observation(payload=True, fingers=(.01, -.01))
    refs = prediction(payload=True)["visual_references"]
    monitor = RoleAwareVisualMonitorV10()
    monitor.configure(views(obs), refs, obs.state, intent_id=1)
    for n in range(1, 5):
        obs = observation((n*.001, 0., 0.), payload=True, fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.attachment.had_observed_evidence
    for n in range(1, 61):
        obs = observation((.004+n*.0002, 0., 0.), payload=True, fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.01, 0, 0, 0, 0, 0, 0])
        if n % 5 == 0:
            retained = copy.deepcopy(refs); retained["external"].update(update="retain", seed=None)
            monitor.configure(views(obs), retained, obs.state, intent_id=n+2)
        cap, reason = monitor.attachment_control_limit()
        assert reason is None and (cap is None or cap >= .001)
    assert monitor.attachment.samples >= 4


def test_retain_does_not_renew_adaptive_attachment_debt_and_rename_cannot_keep_support():
    pixels = rgb(textured()); images = {"external": pixels, "wrist": pixels}
    state = np.zeros(8); monitor = RoleAwareVisualMonitorV10()
    monitor.configure(images, references(payload=True), state, intent_id=1)
    monitor.attachment = supported()
    advance(monitor.attachment, (0., 0.), (0., 0.), distance=.0042)
    monitor._sync_attachment()
    monitor.configure(images, references("retain", payload=True), state, intent_id=2)
    assert monitor.attachment_control_limit() == (0., AdaptiveAttachmentEvidence.handoff_reason)
    assert monitor.attachment.uncertain_travel_m == pytest.approx(.0042)
    renamed = references("retain", payload=True); renamed["external"]["moving_id"] = "other"
    monitor.configure(images, renamed, state, intent_id=3)
    assert monitor.attachment_control_limit()[0] == 0 and not monitor.attachment.had_observed_evidence


def test_previously_predictive_reference_becoming_static_still_contradicts():
    obs = observation(fingers=(.01, -.01)); monitor = RoleAwareVisualMonitorV10()
    monitor.configure(views(obs), prediction()["visual_references"], obs.state, intent_id=1)
    for n in range(1, 11):
        obs = observation((.001*n, 0., 0.), fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.response_ready
    for n in range(1, 11):
        state = np.asarray(obs.state); state[0] += .001*n
        monitor.observe(views(obs), state, [.1, 0, 0, 0, 0, 0, 0])
        if monitor.control_limit()[0] == 0:
            break
    assert monitor.control_limit() == (0., monitor.contradiction_reason)


def test_same_payload_reference_can_recover_after_later_real_grasp():
    obs = observation(payload=True, fingers=(.01, -.01))
    monitor = RoleAwareVisualMonitorV10()
    refs = prediction(payload=True)["visual_references"]
    monitor.configure(views(obs), refs, obs.state, intent_id=1)
    frozen, origin = views(obs), np.asarray(obs.state)
    for n in range(1, 16):
        state = origin.copy(); state[0] += .001*n
        monitor.observe(frozen, state, [.1, 0, 0, 0, 0, 0, 0])
    before = monitor.summary()["reference_health"]
    assert monitor.control_limit()[0] == 0
    retained = copy.deepcopy(refs); retained["external"].update(update="retain", seed=None)
    monitor.configure(frozen, retained, state, intent_id=2)
    for n in range(1, 19):
        moved = observation((.001*n, 0., 0.), payload=True, fingers=(.01, -.01))
        state = np.asarray(moved.state); state[0] += .015
        monitor.observe(views(moved), state, [.1, 0, 0, 0, 0, 0, 0])
        if n <= 3:
            assert monitor.control_limit()[0] == 0
        if monitor.response_ready:
            break
    after = monitor.summary()["reference_health"]
    assert monitor.response_ready and monitor.attachment_control_limit()[1] is None
    assert after["evidence_based_recoveries"] == 1
    assert after["correspondence_id"] == before["correspondence_id"]
    assert after["unvalidated_travel_m"] == 0.


def replay_v9_interval(start, stop):
    path = Path(__file__).resolve().parents[1]/"runs/visual_field_v9_zero_shot_001/trials/bowl_original_zero_hybrid_v9/trace.jsonl"
    if not path.exists():
        pytest.skip("Optional recorded V9 RGB trace absent")
    monitor = RoleAwareVisualMonitorV10(); before = None; result = {}
    def images(value):
        return {"external": np.asarray(decode_image(value["observation/image"])),
                "wrist": np.asarray(decode_image(value["observation/wrist_image"]))}
    for line in path.open():
        row = json.loads(line)
        if row.get("step", 0) > stop:
            break
        if row["event"] in ("reset", "executed") and row.get("step", -1) < start:
            before = row["observation"]
        elif row["event"] == "action" and row["step"] >= start and row["diagnostics"].get("prediction"):
            monitor.configure(images(before), row["diagnostics"]["prediction"]["visual_references"],
                              before["observation/state"], intent_id=row["diagnostics"]["chunk_index"])
        elif row["event"] == "executed" and row["step"] >= start:
            before = row["observation"]
            result[row["step"]] = monitor.observe(images(before), before["observation/state"], row["action"])
    return result


def test_recorded_v9_first_grasp_does_not_gain_false_attachment():
    rows = replay_v9_interval(345, 359)
    assert all(not r["attachment_co_motion_observed"] for r in rows.values())
    assert not rows[359]["attachment"]["had_observed_evidence"]


def test_recorded_v9_slow_carry_reuses_fresh_attachment_evidence():
    rows = replay_v9_interval(531, 621)
    assert any(r["attachment_co_motion_observed"] for step, r in rows.items() if step < 550)
    # Conditional image/state replay, not a claim that new commands would
    # reproduce this path or finish the task. Check actual useful monitor reuse.
    useful = [step for step, r in rows.items() if step >= 541 and
              r["attachment"]["transport_limit_reason"] is None]
    assert len(useful) >= 30
    assert all(r["attachment"]["uncertain_measured_travel_m"] <= r["attachment"]["uncertain_travel_limit_m"] <= .015
               for r in rows.values() if r["attachment"]["transport_limit_reason"] is None)
    # Stronger causal check: the existing budget must actually permit every
    # transition up to the next fresh support, not credit evidence obtained
    # later in the archive after this monitor would already have stopped.
    assert rows[539]["attachment_co_motion_observed"]
    assert .006214 < rows[539]["attachment"]["uncertain_travel_limit_m"] <= .015
    assert all(rows[step]["attachment"]["transport_limit_reason"] is None for step in range(539, 550))
    assert rows[549]["attachment_co_motion_observed"]
    assert rows[549]["attachment"]["supporting_windows"] >= 2
