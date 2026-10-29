"""Response uncertainty, actual contradiction and persistent travel allowance."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import decode_image
from gpt_field.visual_monitor_v8 import RoleAwareVisualMonitorV8
from test_visual_field_v6_policy import observation, prediction
from test_visual_monitor_v5 import rgb, textured
from test_visual_monitor_v6 import references


def views(obs):
    return {"external": np.asarray(decode_image(obs.image)), "wrist": np.asarray(decode_image(obs.wrist_image))}


def moving_monitor(payload=False):
    obs = observation(payload=payload, fingers=(.01, -.01))
    refs = prediction(payload=payload)["visual_references"]
    monitor = RoleAwareVisualMonitorV8()
    monitor.configure(views(obs), refs, obs.state, intent_id=1)
    for n in range(1, 11):
        obs = observation((n*.001, 0., 0.), payload=payload, fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.response_ready
    return monitor, refs, obs


def test_initial_stationary_reference_is_still_rejected_without_budget_renewal():
    pixels = rgb(textured()); images = {"external": pixels, "wrist": pixels}
    state = np.zeros(8); monitor = RoleAwareVisualMonitorV8()
    monitor.configure(images, references(), state, intent_id=1)
    assert monitor.control_limit() == (.005, None)
    for n in range(1, 9):
        state[0] = .002*n
        monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.control_limit()[0] == 0
    assert not monitor.response_ready
    used = monitor.summary()["reference_health"]["unvalidated_travel_m"]
    for mode in ["retain", "seed"]:
        monitor.configure(images, references(mode), state, intent_id=20)
        assert monitor.control_limit()[0] == 0
        assert monitor.summary()["reference_health"]["unvalidated_travel_m"] == used


def test_weak_precision_does_not_call_a_correct_reference_wrong_or_reset_its_fit(monkeypatch):
    monitor, refs, obs = moving_monitor()
    # Inject a poorer reported measurement precision into the evidence layer;
    # actual local feature identities and RGB movement continue unchanged.
    monkeypatch.setattr(monitor, "_noise", lambda track: 3.)
    epoch = monitor.epoch
    for n in range(11, 15):
        obs = observation((n*.001, 0., 0.), fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
        cap, reason = monitor.control_limit()
        assert 0 < cap <= .005 and reason is None
    summary = monitor.summary()
    assert summary["response_evidence"]["state"] == "weakly_observable"
    assert not summary["reference_health"]["repair_required"]
    assert "Do not yaw, retreat or reseed" in summary["reference_health"]["recommended_action"]
    assert summary["reference_health"]["unvalidated_travel_m"] == pytest.approx(.004)
    assert monitor.response.matrix is not None and monitor.epoch == epoch
    refs["external"].update(update="retain", seed=None)
    monitor.configure(views(obs), refs, obs.state, intent_id=2)
    assert monitor.summary()["reference_health"]["unvalidated_travel_m"] == pytest.approx(.004)


def test_weak_response_uses_existing_finite_allowance_and_cannot_be_reseeded_away(monkeypatch):
    monitor, refs, obs = moving_monitor()
    monkeypatch.setattr(monitor, "_noise", lambda track: 3.)
    for n in range(11, 26):
        obs = observation((n*.001, 0., 0.), fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
        keep = copy.deepcopy(refs); keep["external"].update(update="retain", seed=None)
        monitor.configure(views(obs), keep, obs.state, intent_id=n)
    cap, reason = monitor.control_limit()
    assert cap == 0 and reason == monitor.uncertainty_budget_reason
    health = monitor.summary()["reference_health"]
    assert health["repair_required"]
    assert "inconclusive" in health["recommended_action"]
    current = monitor.tracks["moving"].uv.tolist()
    refs["external"]["seed"]["moving_point_uv"] = current
    monitor.configure(views(obs), refs, obs.state, intent_id=40)
    assert monitor.control_limit() == (0., monitor.uncertainty_budget_reason)


def test_useful_stronger_response_restores_reuse_without_new_seed(monkeypatch):
    monitor, refs, obs = moving_monitor()
    monkeypatch.setattr(monitor, "_noise", lambda track: 3.)
    ident = monitor.summary()["reference_health"]["correspondence_id"]
    for mm in [11, 12, 13, 16, 19, 22, 25]:
        obs = observation((mm*.001, 0., 0.), fingers=(.01, -.01))
        monitor.observe(views(obs), obs.state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.response_ready
    assert monitor.control_limit() == (None, None)
    health = monitor.summary()["reference_health"]
    assert health["correspondence_id"] == ident
    assert health["unvalidated_travel_m"] == 0


def test_previously_moving_feature_becoming_static_is_a_predictive_contradiction():
    monitor, _, obs = moving_monitor()
    frozen = views(obs)
    for n in range(1, 11):
        state = np.array(obs.state); state[0] += .001*n
        monitor.observe(frozen, state, [.1, 0, 0, 0, 0, 0, 0])
        if monitor.control_limit()[0] == 0:
            break
    assert monitor.control_limit() == (0., monitor.contradiction_reason)
    assert monitor.summary()["response_evidence"]["state"] == "contradicted"


def test_independent_payload_attachment_and_immediate_open_invalidation_are_preserved():
    monitor, _, obs = moving_monitor(payload=True)
    assert monitor.attachment_evidence
    assert monitor.attachment_control_limit() == (None, None)
    monitor.observe(views(obs), obs.state, [0, 0, 0, 0, 0, 0, -1])
    assert not monitor.attachment_evidence
    assert monitor.attachment_control_limit()[0] == 0


def replay_archive(run, trial, stop):
    path = Path(__file__).resolve().parents[1]/"runs"/run/"trials"/trial/"trace.jsonl"
    if not path.exists():
        pytest.skip("Optional RGB/state rollout archive is absent")
    monitor = RoleAwareVisualMonitorV8(); before = None; rows = {}; configured = {}
    def images(value):
        return {"external": np.asarray(decode_image(value["observation/image"])),
                "wrist": np.asarray(decode_image(value["observation/wrist_image"]))}
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("step", 0) > stop:
                break
            if event["event"] == "reset":
                before = event["observation"]
            elif event["event"] == "action" and event["diagnostics"].get("prediction"):
                value = event["diagnostics"]["prediction"]
                monitor.configure(images(before), value["visual_references"], before["observation/state"],
                                  intent_id=event["diagnostics"]["chunk_index"])
                configured[event["step"]] = monitor.summary()
            elif event["event"] == "executed":
                before = event["observation"]
                summary = monitor.observe(images(before), before["observation/state"], event["action"])
                rows[event["step"]] = (monitor.control_limit(), summary)
    return rows, configured


def test_real_v7_housing_response_is_inconclusive_not_rejected_at_339_and_367():
    # Conditional observation replay only: no simulator/action execution and
    # no counterfactual task success or speed claim from archived observations.
    rows, _ = replay_archive("visual_field_v7_zero_shot_001", "bowl_original_zero_hybrid_v7", 367)
    for step in [339, 367]:
        limit, summary = rows[step]
        assert limit[0] > 0 and limit[1] is None
        assert summary["response_evidence"]["state"] == "weakly_observable"
        assert not summary["reference_health"]["repair_required"]
        assert summary["reference_health"]["unvalidated_travel_m"] < .003
    assert sum(rows[step][1]["response_validated"] for step in range(318, 339)) >= 15


def test_real_v6_stationary_base_still_rejects_and_housing_still_reuses():
    rows, configured = replay_archive("visual_field_v6_zero_shot_001", "bowl_original_zero_hybrid_v6", 188)
    first = next(step for step, (limit, _) in rows.items() if limit[0] == 0)
    assert first < 10
    assert rows[first][1]["reference_health"]["unvalidated_travel_m"] < .01
    assert configured[162]["reference_health"]["repair_required"]
    assert not configured[168]["reference_health"]["repair_required"]
    assert all(rows[step][0][0] != 0 for step in range(168, 189))


@pytest.mark.parametrize("payload", [False, True])
def test_fresh_post_rejection_motion_can_still_rehabilitate_same_reference(monkeypatch, payload):
    import test_visual_monitor_v7 as previous
    monkeypatch.setattr(previous, "RoleAwareVisualMonitorV7", RoleAwareVisualMonitorV8)
    previous.test_fresh_direct_motion_can_rehabilitate_same_previously_stationary_correspondence(payload)
