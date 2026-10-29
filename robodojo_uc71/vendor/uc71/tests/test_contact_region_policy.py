"""Offline compact-region policy contracts; no live model or simulator calls."""
import copy
import json

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.contact_region_policy import (CONTACT_REGION_POLICY_VERSION,
    ChatGPTPlanContactRegionClient, ContactRegionPolicy, contact_region_policy_schema)
from gpt_field.speed_first_v3 import FINE_GRID
from gpt_field.speed_first_v4 import ChatGPTPlanSpeedFirstV4Client, speed_first_v4_schema
from test_field_history import FakeTransport, envelope, observation
from test_speed_first_v2 import tick, records
from test_speed_first_v4 import intent


def region_reference(identifier="region-1", target="plate"):
    view = dict(tool_region_uv=[.15, .3, .35, .5],
        target_region_uv=[.45, .3, .65, .5], desired_offset_uv=[-.02, 0.])
    return dict(reference_id=identifier, target_id=target,
        external=copy.deepcopy(view), wrist=copy.deepcopy(view))


def response(*, direct=False, ref=None, goal=(.03, 0., 0.), identifier=None):
    return intent("pregrasp", direct=direct, goal=goal,
        ref=region_reference() if ref is None else ref, identifier=identifier)


def controller(*responses, apply=True):
    fake = FakeTransport(*(envelope(r, id=f"region-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i, r in enumerate(responses)))
    client = ChatGPTPlanContactRegionClient(token_provider=lambda: "offline-region", transport=fake)
    return ContactRegionPolicy(client, apply_contact_correction=apply), fake


class FeedbackFixture:
    """A candidate source for testing executor safety, not vision efficacy."""
    maximum_orientation_change_rad = .035

    def __init__(self, change=(0., .004, 0.)):
        self.change = np.array(change)
        self.available = True
        self.mismatch = False
        self.applied = []
        self.observed = 0

    def snapshot(self):
        return dict(status="prediction_mismatch" if self.mismatch else "tracking",
            tracking_valid=self.available, response_validated=not self.mismatch,
            restore_nominal_requested=self.mismatch, proposal_available=self.available and not self.mismatch,
            active_views=["external"], pixel_error_norm=8., reference_id="region-1")

    def configure(self, observation, reference, **kwargs):
        return self.snapshot()

    def begin_nominal_goal(self, nominal, key):
        return self.snapshot()

    def notify_nominal_restored(self, nominal):
        return self.snapshot()

    def observe(self, after):
        self.observed += 1
        return self.snapshot()

    def corrected_goal(self, observation, nominal):
        return (np.asarray(nominal) + self.change if self.available else None), self.snapshot()

    def notify_correction_applied(self, observation, nominal, goal):
        self.applied.append((np.asarray(nominal).copy(), np.asarray(goal).copy()))

    def invalidate(self, reason):
        self.available = False
        return self.snapshot()


def effective_goal(policy):
    return np.asarray(policy.field.center) + np.asarray(policy._field_cache.effective_compact_field["goal_offset_m"])


def test_new_schema_and_prompt_are_opt_in_without_legacy_contact_requirements():
    old = ChatGPTPlanSpeedFirstV4Client(token_provider=lambda: "offline")
    baseline = old.build_payload(observation().to_dict())
    new = ChatGPTPlanContactRegionClient(token_provider=lambda: "offline")
    payload = new.build_payload(observation().to_dict())
    schema = payload["text"]["format"]["schema"]
    assert schema == contact_region_policy_schema()
    assert baseline["text"]["format"]["schema"] == speed_first_v4_schema()
    assert old.build_payload(observation().to_dict()) == baseline
    region = schema["properties"]["visual_intent"]["properties"]["pregrasp_reference"]
    assert "tool_region_uv" in json.dumps(region)
    assert "tool_contact_uv" not in json.dumps(region)
    assert "tool_contact_uv" not in payload["instructions"]
    assert "24pixels" not in payload["instructions"]
    assert "IMAGE ANCHORS" in payload["instructions"]
    assert "never automatically zero" in payload["instructions"]
    assert "EXACT SAME reference_id" in payload["instructions"]
    assert "conditional rigid-hand assumption" in payload["instructions"]
    assert "target regions still require visual tracking" in payload["instructions"]
    assert new.prompt_version.startswith(CONTACT_REGION_POLICY_VERSION)
    assert payload["reasoning"] == {"effort": "medium"}


def test_apply_and_shadow_have_identical_first_payload_and_direct_actions():
    raw = response(direct=True)
    raw["direct"]["actions"] = [[.02*i, -.01, .01, 0., 0., 0., -1.] for i in range(5)]
    on, ton = controller(raw, apply=True)
    off, toff = controller(raw, apply=False)
    on.pregrasp, off.pregrasp = FeedbackFixture(), FeedbackFixture()
    obs_on = obs_off = observation()
    for expected in raw["direct"]["actions"]:
        obs_on, a, _ = tick(on, obs_on)
        obs_off, b, _ = tick(off, obs_off)
        assert a.action == b.action == tuple(expected)
        assert a.diagnostics["execution_kind"] == b.diagnostics["execution_kind"] == "direct_five_actions"
    assert ton.calls == toff.calls
    assert on.pregrasp.observed == off.pregrasp.observed == 5
    assert not on.pregrasp.applied and not off.pregrasp.applied
    assert on.contact_region_stats()["applied_goals"] == off.contact_region_stats()["applied_goals"] == 0


def test_region_wire_history_and_ledger_preserved_after_private_adapter():
    raw = response(direct=True)
    policy, fake = controller(raw, response(direct=True))
    policy.pregrasp = FeedbackFixture()
    obs = observation()
    for _ in range(6):
        obs, _, _ = tick(policy, obs)
    assert policy.client.last_compact_prediction == raw
    assert policy.client.last_visual_intent["pregrasp_reference"] == raw["visual_intent"]["pregrasp_reference"]
    history = records(policy.client.history_snapshot())
    assert raw in history
    assert raw in records(fake.calls[1]["input"])
    assert "Region handled by the separate" not in json.dumps(fake.calls)
    ledger = policy.client.operation_ledger.presentation(2, current_state=obs.state)
    current = ledger["recent_operations"][-1]
    assert current["visual_reference_supplied"] is True
    assert current["visual_unavailability"] == {"external": None, "wrist": None}
    assert current["contact_region_reference"]["reference_id"] == "region-1"
    assert current["contact_region_reference"]["available_views"] == ["external", "wrist"]


@pytest.mark.parametrize("mutation", [
    lambda r: r["visual_intent"]["pregrasp_reference"]["external"].update(tool_region_uv=[.8, .2, .1, .5]),
    lambda r: r["visual_intent"]["pregrasp_reference"].update(target_id="wrong"),
    lambda r: r["visual_intent"]["unavailable_reasons"].update(external="But provided"),
    lambda r: r["operation"].update(kind="move"),
    lambda r: r["direct"]["actions"][0].__setitem__(0, 10.),
    lambda r: r["direct"]["actions"][0].__setitem__(6, 1.),
])
def test_invalid_response_rejected_without_diagnostic_or_history_mutation(mutation):
    client = ChatGPTPlanContactRegionClient(token_provider=lambda: "offline")
    raw = response(direct=True)
    client.validate_response(raw, FINE_GRID)
    previous = copy.deepcopy((client.last_operation, client.last_visual_intent,
        client.last_pregrasp_reference, client.last_compact_prediction, client.history_snapshot()))
    bad = copy.deepcopy(raw)
    mutation(bad)
    with pytest.raises(SchemaError):
        client.validate_response(bad, FINE_GRID)
    assert (client.last_operation, client.last_visual_intent, client.last_pregrasp_reference,
        client.last_compact_prediction, client.history_snapshot()) == previous


def test_apply_uses_immutable_bounded_nominal_shadow_keeps_original_execution():
    raw = response()
    on, ton = controller(raw)
    off, toff = controller(raw, apply=False)
    on.pregrasp, off.pregrasp = FeedbackFixture(), FeedbackFixture()
    obs_on = obs_off = observation()
    for _ in range(4):
        obs_on, _, _ = tick(on, obs_on, fraction=.05)
        obs_off, _, _ = tick(off, obs_off, fraction=.05)
    assert len(ton.calls) == len(toff.calls) == 1
    assert ton.calls[0] == toff.calls[0]
    np.testing.assert_allclose(effective_goal(on), on._nominal_goal + [0., .004, 0.])
    np.testing.assert_allclose(effective_goal(off), off._nominal_goal)
    assert on.pregrasp.applied and not off.pregrasp.applied
    assert off.contact_region_stats()["shadow_proposals"] > 0
    assert off.contact_region_stats()["applied_goals"] == 0
    assert off._pregrasp_alignment_active() is False
    assert off._field_policy.tracker.endpoint_tolerance_m == .0025
    for nominal, goal in on.pregrasp.applied:
        np.testing.assert_allclose(nominal, on._nominal_goal)
        assert np.linalg.norm(goal - nominal) <= .006


@pytest.mark.parametrize("failure", ["mismatch", "lost"])
def test_prediction_mismatch_or_loss_restores_nominal_before_next_action(failure):
    p, transport = controller(response())
    p.pregrasp = FeedbackFixture()
    obs, _, _ = tick(p, observation(), fraction=.05)
    assert p._pregrasp_goal_active
    if failure == "mismatch":
        p.pregrasp.mismatch = True
    else:
        p.pregrasp.available = False
    # A completed ordinary transition reports the problem and restores cached goal.
    obs, _, _ = tick(p, obs, fraction=.05)
    np.testing.assert_allclose(effective_goal(p), p._nominal_goal)
    assert not p._pregrasp_goal_active
    assert p.contact_region_stats()["nominal_goal_restorations"] == 1
    tick(p, obs, fraction=.05)
    assert p._field_policy.tracker.endpoint_tolerance_m == .0025
    assert len(transport.calls) == 1


@pytest.mark.parametrize("change", [(0., .0061, 0.), (float("nan"), 0., 0.)])
def test_bad_proposed_goal_cannot_change_nominal(change):
    p, _ = controller(response())
    p.pregrasp = FeedbackFixture(change)
    tick(p, observation(), fraction=.05)
    np.testing.assert_allclose(effective_goal(p), p._nominal_goal)
    assert p.contact_region_stats()["applied_goals"] == 0
    assert p.contact_region_stats()["rejected_proposals"] > 0


def test_bad_feedback_boundary_is_atomic_for_region_observer():
    from dataclasses import replace
    p, _ = controller(response())
    p.pregrasp = FeedbackFixture()
    obs = observation()
    d = p.act(obs)
    initial = p.contact_region_stats()
    with pytest.raises(ValueError, match="exact pending"):
        p.observe_transition(replace(obs, state=(.2, *obs.state[1:])), d, obs)
    assert p.contact_region_stats() == initial


def test_apply_flag_must_be_explicit_boolean():
    client = ChatGPTPlanContactRegionClient(token_provider=lambda: "offline")
    with pytest.raises(ValueError):
        ContactRegionPolicy(client, apply_contact_correction="off")


def rgb_fixture():
    from dataclasses import replace
    from test_speed_first_pregrasp import sample, CENTERS, SIZE

    def frame(x, target_shift=(0., 0.), tool_extra=(0., 0.)):
        value, _ = sample((x, 0., 0.), external_target_shift=target_shift, tool_extra=tool_extra)
        return replace(value, prompt=observation().prompt)

    tool, target = (np.asarray(CENTERS["external_" + role]) for role in ("tool", "target"))
    ref = dict(reference_id="synthetic-regions", target_id="black_bowl", wrist=None,
        external=dict(tool_region_uv=(np.r_[tool-16, tool+16]/(SIZE-1)).tolist(),
            target_region_uv=(np.r_[target-16, target+16]/(SIZE-1)).tolist(),
            desired_offset_uv=(np.array([-20., -22.])/(SIZE-1)).tolist()))
    raw = response(ref=ref)
    raw["visual_intent"]["unavailable_reasons"]["wrist"] = "Only external synthetic regions supplied"
    return frame, raw


@pytest.mark.parametrize("apply,defect", [(True, "target_motion"), (True, "response_mismatch"), (False, None)])
def test_actual_rgb_region_core_learns_and_reaches_policy_goal_path(apply, defect):
    """Exercise texture flow -> held-out response -> real Field target update.

    Synthetic observations test plumbing and geometry only, not actual grasp
    performance. Both arms receive the same prescribed measured trajectory.
    """
    frame, raw = rgb_fixture()
    policy, transport = controller(raw, apply=apply)
    before = frame(0.)
    for index in range(1, 9):
        decision = policy.act(before)
        after = frame(.002*index)
        policy.observe_transition(before, decision, after)
        before = after
    stats = policy.contact_region_stats()
    assert stats["references_seeded"] == 1, stats
    assert stats["tracked_transitions"] == 8, stats
    assert stats["core"]["held_out_samples"] >= 2, stats
    assert stats["core"]["response_validated"], stats
    assert stats["accepted_proposals"] > 0, stats
    assert len(transport.calls) == 1
    assert stats["applied_goals"] > 0 if apply else stats["applied_goals"] == 0
    if apply:
        assert 0. < np.linalg.norm(effective_goal(policy)-policy._nominal_goal) <= .006 + 1e-12
        # A true RGB observation inconsistency must revoke the cached path.
        decision = policy.act(before)
        after = (frame(.018, target_shift=(5., 0.)) if defect == "target_motion"
                 else frame(.018, tool_extra=(-7., 0.)))
        policy.observe_transition(before, decision, after)
        np.testing.assert_allclose(effective_goal(policy), policy._nominal_goal)
        assert not policy._pregrasp_goal_active
        assert policy.contact_region_stats()["nominal_goal_restorations"] >= 1
        if defect == "response_mismatch":
            assert policy.contact_region_stats()["prediction_mismatch_events"] >= 1
    else:
        np.testing.assert_allclose(effective_goal(policy), policy._nominal_goal)
        assert not policy._pregrasp_alignment_active()


def test_new_accepted_nominal_goal_retains_real_fit_without_restoring_old_target():
    frame, first = rgb_fixture()
    first["operation"]["local_action_budget"] = 8
    second = copy.deepcopy(first)
    second["field"]["goal_offset_m"] = [.01, 0., 0.]
    policy, transport = controller(first, second)
    before = frame(0.)
    for index in range(1, 9):
        d = policy.act(before)
        after = frame(.002*index)
        policy.observe_transition(before, d, after)
        before = after
    previous = policy.pregrasp.snapshot()
    assert previous["response_validated"]
    assert policy._pregrasp_goal_active
    old_nominal = policy._nominal_goal.copy()
    assert policy.pending_field_event["reason"] == "field_budget_exhausted"
    policy.act(before)
    assert len(transport.calls) == 2
    current = policy.pregrasp.snapshot()
    assert current["epoch"] == previous["epoch"]
    assert current["held_out_samples"] == previous["held_out_samples"]
    assert current["response_validated"]
    assert not current["restore_nominal_requested"]
    np.testing.assert_allclose(policy._nominal_goal, np.asarray(before.state[:3]) + [.01, 0., 0.])
    assert not np.allclose(old_nominal, policy._nominal_goal)
    np.testing.assert_allclose(effective_goal(policy), policy._nominal_goal)
