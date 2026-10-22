"""Physical-direction recovery, actual RGB evidence and exact dry admission.

These tests use measured synthetic robot transitions and tracked rendered RGB.
They never use depth, environment poses, collision truth or model inference.
"""
import copy
import pickle

import numpy as np
import pytest

from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v5 import LocalFeatureBundle
from gpt_field.visual_translation_guard_v15 import ObservedRiskTranslationGuardV15
from gpt_field.robot_translation_response import MeasuredTranslationResponse
from test_visual_translation_guard_v11 import plant as old_plant, pixels, context
from test_visual_translation_guard_v14 import assess, observe, consume, configure_mode


def plant(mode="transit", *, tool=(43, 53, 125, 77), small_support=False):
    ctx = context(mode=mode, tool=tool)
    _, monitor, state = old_plant(ctx)
    guard = ObservedRiskTranslationGuardV15()
    im = pixels()
    if small_support:
        monitor.tracks["moving"] = LocalFeatureBundle(_gray(im), [55/223, 65/223],
            "small_visible_tool_part", support_radius_px=4., maximum_features=3,
            roi_uv=np.array([43, 53, 67, 77])/223)
    guard.configure({"external": im, "wrist": im}, ctx, state, target_id="target", monitor=monitor)
    return guard, monitor, state


def reseed(guard, monitor, state, *, shift=12, name="new_visible_tool_part"):
    image = pixels(shift)
    monitor.tracks["moving"] = LocalFeatureBundle(_gray(image), [(55+shift)/223, 65/223],
        name, roi_uv=np.array([43+shift, 53, 67+shift, 77])/223)
    guard.configure({"external": image, "wrist": image},
                    context(mode="separate", tool=(30, 45, 150, 85)), state,
                    target_id="target", monitor=monitor)
    assert not guard.context_invalid
    return image


def test_new_reference_reuses_unfinished_heading_without_spending_slot_or_refilling():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [0, 1, 0], 4)
    state = consume(guard, monitor, state, [0, 0, 1], 1)
    before = copy.deepcopy(guard.experiments)
    debt = guard.observation_travel_m
    revisions = len(guard._support_revision_events)
    reseed(guard, monitor, state)
    assert guard._support_generation == 1
    assert guard.experiments == before
    for _ in range(4):
        choice = assess(guard, monitor, [0, 0, .002])
        assert choice["accepted"]
        assert guard._experiment_pending["index"] == 1
    assert len(guard._support_revision_events) == revisions
    assert len(guard.experiments) == 2 and guard.observation_travel_m == debt
    state = observe(guard, monitor, state, [0, 0, .002], [0, 0, .001], pixel_x=12)
    assert len(guard.experiments) == 2
    assert guard.experiments[1]["measured_travel_m"] == pytest.approx(.002)
    assert guard.observation_travel_m == pytest.approx(.006)
    assert len(guard._support_revision_events) == revisions+1
    assert len(guard.experiments[1]["support_revisions"]) == 2
    # Earlier +Y is still available, not lost behind the most recent +Z row.
    choice = assess(guard, monitor, [0, .002, 0])
    assert choice["accepted"] and guard._experiment_pending["index"] == 0
    state = observe(guard, monitor, state, [0, .002, 0], [0, .001, 0], pixel_x=12)
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.005)
    assert guard.experiments[1]["measured_travel_m"] == pytest.approx(.002)
    assert guard.observation_travel_m == pytest.approx(.007)


def test_checkpoint_enables_distinct_direction_early_but_not_an_unreviewed_turn():
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 3)
    assert assess(guard, monitor, [0, 0, .002])["reason"] == "translation_new_experiment_requires_nominal_separation"
    configure_mode(guard, monitor, state)
    assert assess(guard, monitor, [0, 0, .002])["reason"] == "translation_recovery_direction_change_requires_review"
    state = consume(guard, monitor, state, [1, 0, 0], 1)
    assert not assess(guard, monitor, [0, 0, .002], nominal_away=False)["accepted"]
    choice = assess(guard, monitor, [0, 0, .002])
    assert choice["accepted"]
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.004)
    state = observe(guard, monitor, state, [0, 0, .002], [0, 0, .001])
    configure_mode(guard, monitor, state, "transit")
    assert assess(guard, monitor, [.002, 0, 0])["reason"] == "translation_direction_resume_requires_nominal_separation"
    configure_mode(guard, monitor, state)
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]


def test_actual_support_loss_allows_reviewed_new_heading_but_reseed_alone_does_not():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 1)
    reseed(guard, monitor, state)
    assert assess(guard, monitor, [0, 0, .002])["reason"] == "translation_recovery_direction_change_requires_review"
    # A real missing feature in a newly observed frame is different from a
    # model naming another current feature or merely changing its identifier.
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    missing = pixels(missing=True)
    monitor.tracks["moving"].track(_gray(missing))
    after = state.copy();after[0] += .0005
    guard.observe({"external": missing, "wrist": missing}, after,
                  [.04, 0, 0, 0, 0, 0, 0], monitor=monitor)
    assert guard.experiments[0]["support_loss_observed"]
    assert not assess(guard, monitor, [0, 0, .002])["accepted"]  # Repair still required.
    reseed(guard, monitor, after, shift=0)
    assert assess(guard, monitor, [0, 0, .002])["accepted"]
    assert len(guard.experiments) == 1


def test_exhausted_heading_never_renews_for_large_reference_change_or_new_id():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 8)
    reseed(guard, monitor, state)
    for degrees in (0, 15, 29, 35, 44):
        theta = np.deg2rad(degrees)
        choice = assess(guard, monitor, [.002*np.cos(theta), .002*np.sin(theta), 0])
        assert not choice["accepted"]
    assert len(guard.experiments) == 1
    assert guard.observation_travel_m == pytest.approx(.008)


def test_three_distinct_directions_remain_finite_and_old_heading_can_resume():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 4)
    state = consume(guard, monitor, state, [0, 0, 1], 4)
    state = consume(guard, monitor, state, [0, 1, 0], 4)
    assert assess(guard, monitor, [-.002, 0, 0])["reason"] == "translation_recovery_experiment_limit"
    state = consume(guard, monitor, state, [1, 0, 0], 4)
    state = consume(guard, monitor, state, [0, 0, 1], 4)
    assert guard.observation_travel_m == pytest.approx(.020)
    assert assess(guard, monitor, [0, .002, 0])["reason"] == "translation_observation_budget_exhausted"
    state = observe(guard, monitor, state, [0, 0, 0], [0, 0, .0002])
    assert guard.observation_travel_m == pytest.approx(.0202)
    assert guard.experiments[1]["measured_travel_m"] == pytest.approx(.0082)
    assert len(guard.experiments) == 3


def test_current_command_fit_reassesses_old_tail_without_rounding_up():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 7)
    assert assess(guard, monitor, [.001, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.001, 0, 0], [.0005, 0, 0])
    assert not assess(guard, monitor, [.002, 0, 0])["accepted"]
    class SlowRobot:
        def command_cap_for_measured_limit(self, direction, limit, requested, state):
            return {"command_cap_m": min(requested, limit/.25)}
    choice = assess(guard, monitor, [.005, 0, 0], translation_response=SlowRobot())
    assert choice["accepted"] and choice["cap_m"] == pytest.approx(.002)
    assert .25*choice["cap_m"] <= .0005+1e-12
    assert assess(guard, monitor, [.002, 0, 0], translation_response=SlowRobot())["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.0004, 0, 0])
    assert not guard.summary()["recovery_experiments"]["trials"][0]["last_command_fit_failed"]
    assert not assess(guard, monitor, [.002, 0, 0], translation_response=SlowRobot())["accepted"]


def test_preview_preserves_pending_command_identity_models_and_all_state():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 4)
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    before = pickle.dumps(guard.__dict__)
    identities = (guard.__dict__, guard.tool_track, guard.neighbors[0]["track"],
                  guard.cumulative_response, guard._experiment_pending)
    for command, kwargs in (([0, 0, .002], {}), ([.002, 0, 0], {"nominal_clear": False}),
                            ([0, 0, 0], {})):
        result = guard.preview_assess(command, monitor=monitor, nominal_clear=kwargs.get("nominal_clear", True),
                                    nominal_away=True, nominal_regions=[{"id": "neighbor", "core_distance_m": .04, "uncertainty_m": .03}])
        assert result["preview"] and not result["requires_explicit_model_choice"]
        assert pickle.dumps(guard.__dict__) == before
        assert identities == (guard.__dict__, guard.tool_track, guard.neighbors[0]["track"],
                              guard.cumulative_response, guard._experiment_pending)
    # The original selection remains the one acknowledged by the next action.
    state = observe(guard, monitor, state, [.002, 0, 0], [.001, 0, 0])
    assert not guard.domain_invalid and len(guard.experiments) == 1
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.005)


def test_preview_exact_decision_and_prospective_mode_do_not_change_live_context():
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 4)
    kwargs = dict(monitor=monitor, nominal_clear=True, nominal_away=True,
                  nominal_regions=[{"id": "neighbor", "core_distance_m": .04, "uncertainty_m": .03}])
    before = pickle.dumps(guard.__dict__)
    actual_mode = guard.preview_assess([0, 0, .002], **kwargs)
    prospective = guard.preview_assess([0, 0, .002], prospective_translation_mode="separate", **kwargs)
    assert not actual_mode["accepted"]
    assert prospective["accepted"] and prospective["requires_explicit_model_choice"]
    assert pickle.dumps(guard.__dict__) == before and guard.context["translation_mode"] == "transit"
    # Commit only the proposed context to compare the exact same live assessment.
    guard.context = {**guard.context, "translation_mode": "separate", "intended_contact_id": None}
    live = guard.assess([0, 0, .002], **kwargs)
    assert {k: v for k, v in prospective.items() if k not in (
        "preview", "requires_explicit_model_choice", "prospective_translation_mode")} == live
    assert len(guard.experiments) == 1  # Assessment, including live, is not execution.


def test_preview_cap_failure_and_exception_restore_every_provisional_write():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 7)
    assert assess(guard, monitor, [.001, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.001, 0, 0], [.0005, 0, 0])
    kwargs = dict(monitor=monitor, nominal_clear=True, nominal_away=True)
    before = pickle.dumps(guard.__dict__)
    result = guard.preview_assess([.002, 0, 0], **kwargs)
    assert result["reason"] == "translation_measured_allowance_below_useful_command_floor"
    assert pickle.dumps(guard.__dict__) == before
    class FailedResponse:
        def command_cap_for_measured_limit(self, *args):
            raise RuntimeError("predictor unavailable")
    with pytest.raises(RuntimeError, match="predictor unavailable"):
        guard.preview_assess([.002, 0, 0], translation_response=FailedResponse(), **kwargs)
    assert pickle.dumps(guard.__dict__) == before


def test_rejected_downstream_hold_preserves_charge_and_does_not_commit_reseed():
    guard, monitor, state = plant("separate")
    state = consume(guard, monitor, state, [1, 0, 0], 1)
    revisions = len(guard._support_revision_events)
    reseed(guard, monitor, state)
    remote = [{"id": "neighbor", "core_distance_m": 1., "uncertainty_m": .01}]
    assert assess(guard, monitor, [.002, 0, 0], nominal_regions=remote)["accepted"]
    state = observe(guard, monitor, state, [0, 0, 0], [.0003, 0, 0], pixel_x=12)
    assert guard.observation_travel_m == pytest.approx(.0013)
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.0013)
    assert len(guard._support_revision_events) == revisions


def test_causal_rgb_response_can_replace_uncertain_experiment_before_eight_mm():
    guard, monitor, state = plant("separate", tool=(43, 10, 125, 210), small_support=True)
    robot = MeasuredTranslationResponse()
    reasons = []
    for step in range(1, 9):
        choice = assess(guard, monitor, [-.002, 0, 0], translation_response=robot)
        assert choice["accepted"], choice["reason"]
        reasons.append(choice["reason"])
        state = observe(guard, monitor, state, [-.002, 0, 0], [-.001, 0, 0], pixel_x=-step, response=robot)
        if choice["reason"] == "projected_separation_conditional_empirical_evidence":
            break
    assert guard.cumulative_response_ready and len(guard._response_error) >= 2
    assert not guard.improving  # Whole-envelope evidence genuinely inconclusive.
    assert reasons[0] == "projected_risk_finite_directional_experiment"
    assert reasons[-1] == "projected_separation_conditional_empirical_evidence"
    assert guard.observation_travel_m < .008 and len(guard.experiments) == 1
    before = pickle.dumps(guard.__dict__)
    preview = guard.preview_assess([-.002, 0, 0], monitor=monitor, nominal_away=True,
        nominal_clear=True, translation_response=robot)
    assert preview["accepted"] and not preview["diagnostics"]["clearance_certified"]
    assert pickle.dumps(guard.__dict__) == before
    guard.worsening.add("neighbor")
    assert assess(guard, monitor, [-.002, 0, 0], translation_response=robot)["reason"] == "translation_observed_neighbor_approach"


@pytest.mark.parametrize("failure", ["geometry", "track", "neighbor", "domain"])
def test_preview_and_recovery_do_not_override_hard_refusals(failure):
    guard, monitor, state = plant("separate")
    kwargs = dict(monitor=monitor, nominal_clear=True, nominal_away=True)
    if failure == "geometry": kwargs["nominal_clear"] = False
    if failure == "track": monitor.tracks["moving"].reliable = False
    if failure == "neighbor": guard.neighbors[0]["track"].reliable = False
    if failure == "domain": guard.domain_invalid = "translation_tool_configuration_changed"
    before = pickle.dumps(guard.__dict__)
    preview = guard.preview_assess([0, 0, .002], prospective_translation_mode="separate", **kwargs)
    assert not preview["accepted"] and pickle.dumps(guard.__dict__) == before
    assert not guard.assess([0, 0, .002], **kwargs)["accepted"]
    assert not guard.experiments
