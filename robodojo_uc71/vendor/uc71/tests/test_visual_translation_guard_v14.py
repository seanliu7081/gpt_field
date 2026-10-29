"""Finite recovery experiments using actual synthetic RGB/robot transitions.

Archive tests inspect only saved model geometry and recorded observations. No
simulator, model inference, evaluator state or camera calibration is used.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v5 import LocalFeatureBundle
from gpt_field.visual_translation_guard_v14 import ObservedRiskTranslationGuardV14
from gpt_field.robot_translation_response import MeasuredTranslationResponse
from test_visual_translation_guard_v11 import plant as old_plant, pixels, context


def plant(mode="transit", *, tool=(43, 53, 125, 77), small_support=False):
    ctx = context(mode=mode, tool=tool)
    _, monitor, state = old_plant(ctx)
    guard = ObservedRiskTranslationGuardV14()
    im = pixels()
    if small_support:
        monitor.tracks["moving"] = LocalFeatureBundle(_gray(im), [55/223, 65/223],
            "small_visible_tool_part", support_radius_px=4., maximum_features=3,
            roi_uv=np.array([43, 53, 67, 77])/223)
    guard.configure({"external": im, "wrist": im}, ctx, state, target_id="target", monitor=monitor)
    return guard, monitor, state


def assess(guard, monitor, command, **kwargs):
    defaults = dict(nominal_clear=True, nominal_away=True,
                    nominal_regions=[{"id": "neighbor", "core_distance_m": .04, "uncertainty_m": .03}])
    defaults.update(kwargs)
    return guard.assess(command, monitor=monitor, **defaults)


def observe(guard, monitor, state, command, actual_delta, *, pixel_x=0, response=None):
    image = pixels(pixel_x)
    monitor.tracks["moving"].track(_gray(image))
    after = state.copy()
    after[:3] += actual_delta
    action = np.r_[np.asarray(command)/.05, np.zeros(4)]
    if response is not None:
        response.observe(state, action, after)
    guard.observe({"external": image, "wrist": image}, after, action, monitor=monitor)
    return after


def consume(guard, monitor, state, direction, count):
    direction = np.asarray(direction, float)
    for _ in range(count):
        choice = assess(guard, monitor, direction*.002)
        assert choice["accepted"], choice["reason"]
        command = direction*min(.002, choice["cap_m"] or .002)
        assert assess(guard, monitor, command)["accepted"]
        state = observe(guard, monitor, state, command, direction*.001)
    return state


def configure_mode(guard, monitor, state, mode="separate"):
    image = pixels()
    guard.configure({"external": image, "wrist": image},
                    context(mode=mode, tool=(43, 53, 125, 77)), state,
                    target_id="target", monitor=monitor)


def test_assess_and_cap_reselection_do_not_commit_or_charge_experiments():
    guard, monitor, state = plant()
    for _ in range(20):
        choice = assess(guard, monitor, [.008, 0, 0])
        assert choice["cap_m"] == pytest.approx(.005)
    assert guard.experiments == [] and guard.observation_travel_m == 0
    # The exact reduced command is rechecked before actual execution.
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.0006, 0, 0])
    assert len(guard.experiments) == 1
    assert guard.experiments[0]["acknowledged_actions"] == 1
    assert guard.observation_travel_m == pytest.approx(.0006)
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.0006)


def test_checkpoint_does_not_stop_but_direction_exhaustion_requires_new_experiment():
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 4)
    row = guard.summary()["recovery_experiments"]["trials"][0]
    assert row["information_checkpoint_reached"] and not row["exhausted"]
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    state = consume(guard, monitor, state, [1, 0, 0], 4)
    assert assess(guard, monitor, [.002, 0, 0])["reason"] == "translation_recovery_direction_exhausted"
    assert guard.observation_travel_m == pytest.approx(.008)
    # Original-mode direction changes cannot create a new allowance.
    assert assess(guard, monitor, [0, 0, .002])["reason"] == "translation_new_experiment_requires_nominal_separation"
    configure_mode(guard, monitor, state)
    assert not assess(guard, monitor, [0, 0, .002], nominal_away=False)["accepted"]
    assert assess(guard, monitor, [0, 0, .002])["accepted"]
    assert len(guard.experiments) == 1
    state = observe(guard, monitor, state, [0, 0, .002], [0, 0, .001])
    assert len(guard.experiments) == 2
    assert guard.observation_travel_m == pytest.approx(.009)


def test_three_direction_experiments_share_twenty_mm_and_holds_never_renew():
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 8)
    configure_mode(guard, monitor, state)
    state = consume(guard, monitor, state, [0, 0, 1], 8)
    state = consume(guard, monitor, state, [0, 1, 0], 3)
    # At19mm, remaining measured1mm caps the command under the cold unit prior.
    choice = assess(guard, monitor, [0, .002, 0])
    assert choice["accepted"] and choice["cap_m"] == pytest.approx(.001)
    assert assess(guard, monitor, [0, .001, 0])["accepted"]
    state = observe(guard, monitor, state, [0, .001, 0], [0, .001, 0])
    assert len(guard.experiments) == 3
    assert assess(guard, monitor, [0, .002, 0])["reason"] == "translation_observation_budget_exhausted"
    hold = assess(guard, monitor, [0, 0, 0])
    assert hold["reason"] == "pose_hold_no_clearance_claim"
    state = observe(guard, monitor, state, [0, 0, 0], [0, .0002, 0])
    assert guard.observation_travel_m == pytest.approx(.0202)
    assert guard.experiments[-1]["measured_travel_m"] == pytest.approx(.0042)
    configure_mode(guard, monitor, state)
    assert not assess(guard, monitor, [-.002, 0, 0])["accepted"]
    assert "cold_separation_remaining_m" not in guard.summary()


def test_submillimeter_measured_tail_can_fit_useful_command_then_switch():
    class SlowPlant:
        def command_cap_for_measured_limit(self, direction, limit, requested, state):
            return {"command_cap_m": min(requested, limit/.25)}
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 7)
    assert assess(guard, monitor, [.001, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.001, 0, 0], [.0005, 0, 0])
    cap = assess(guard, monitor, [.005, 0, 0], translation_response=SlowPlant())
    assert cap["accepted"] and cap["cap_m"] == pytest.approx(.002)
    assert assess(guard, monitor, [.002, 0, 0], translation_response=SlowPlant())["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.0004, 0, 0])
    assert not assess(guard, monitor, [.002, 0, 0], translation_response=SlowPlant())["accepted"]
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.0079)
    configure_mode(guard, monitor, state)
    assert assess(guard, monitor, [0, 0, .002])["accepted"]


def test_small_heading_jitter_and_same_footprint_renaming_do_not_renew():
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 8)
    im = pixels()
    monitor.tracks["moving"] = LocalFeatureBundle(_gray(im), [57/223, 65/223],
        "renamed", roi_uv=np.array([43, 53, 70, 77])/223)
    configure_mode(guard, monitor, state)
    assert not guard.summary()["recovery_experiments"]["support_change"]["qualifies_as_changed_support"]
    for degrees in (0, 15, 29, 35, 44):
        theta = np.deg2rad(degrees)
        assert not assess(guard, monitor, [.002*np.cos(theta), .002*np.sin(theta), 0])["accepted"]
    assert len(guard.experiments) == 1 and guard.observation_travel_m == pytest.approx(.008)


def test_changed_physical_support_permits_one_new_experiment_without_erasing_debt():
    guard, monitor, state = plant()
    state = consume(guard, monitor, state, [1, 0, 0], 8)
    # A distinct current feature at least8px away is a model reference revision,
    # not observed motion and not a certificate of semantic identity.
    im = pixels(12)
    monitor.tracks["moving"] = LocalFeatureBundle(_gray(im), [67/223, 65/223],
        "other_tool_feature", roi_uv=np.array([55, 53, 79, 77])/223)
    guard.configure({"external": im, "wrist": im}, context(mode="separate", tool=(43, 53, 125, 77)),
                    state, target_id="target", monitor=monitor)
    assert guard.summary()["recovery_experiments"]["support_change"]["qualifies_as_changed_support"]
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.001, 0, 0], pixel_x=12)
    assert len(guard.experiments) == 2 and guard.observation_travel_m == pytest.approx(.009)
    for _ in range(3):
        guard.configure({"external": im, "wrist": im}, context(mode="separate", tool=(43, 53, 125, 77)),
                        state, target_id="target", monitor=monitor)
        assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    assert len(guard.experiments) == 2


def test_rejected_reselection_executes_hold_without_starting_provisional_experiment():
    guard, monitor, state = plant()
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    assert not assess(guard, monitor, [.002, 0, 0], nominal_clear=False)["accepted"]
    observe(guard, monitor, state, [0, 0, 0], [0, 0, 0])
    assert guard.experiments == []


def test_downstream_hold_after_provisional_free_command_preserves_uncertain_debt():
    guard, monitor, state = plant()
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.001, 0, 0])
    # New model geometry marks this segment remote, so the provisional Field
    # command is free of projected risk. A later target/payload gate rejects it.
    remote = [{"id": "neighbor", "core_distance_m": 1., "uncertainty_m": .01}]
    result = assess(guard, monitor, [.002, 0, 0], nominal_regions=remote)
    assert result["accepted"] and result["reason"] == "projected_risk_no_near_conflict"
    state = observe(guard, monitor, state, [0, 0, 0], [.0003, 0, 0])
    assert guard.observation_travel_m == pytest.approx(.0013)
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.0013)
    assert len(guard.experiments) == 1


def test_downstream_hold_after_provisional_uncertain_command_does_not_start_debt():
    guard, monitor, state = plant()
    remote = [{"id": "neighbor", "core_distance_m": 1., "uncertainty_m": .01}]
    assert assess(guard, monitor, [.002, 0, 0], nominal_regions=remote)["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.001, 0, 0])
    assert guard.observation_travel_m == 0.
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [0, 0, 0], [.0003, 0, 0])
    assert guard.observation_travel_m == 0.
    assert guard.experiments == []


def learn_separation():
    # The controlled corner bundle is small relative to the whole-tool box.
    # Affine extrapolation therefore has a genuinely larger boundary error:
    # reference response can validate before the full-box gap changes beyond
    # its propagated noise. No readiness/improving flags are manufactured.
    guard, monitor, state = plant("separate", tool=(43, 10, 125, 210), small_support=True)
    robot = MeasuredTranslationResponse()
    reasons = []
    for step in range(1, 9):
        result = assess(guard, monitor, [-.002, 0, 0], translation_response=robot)
        assert result["accepted"], result["reason"]
        reasons.append(result["reason"])
        state = observe(guard, monitor, state, [-.002, 0, 0], [-.001, 0, 0],
                        pixel_x=-step, response=robot)
        if result["reason"] == "projected_separation_conditional_empirical_evidence":
            break
    return guard, monitor, state, robot, reasons


def test_causal_cold_to_conditional_separation_before_eight_mm():
    guard, monitor, state, robot, reasons = learn_separation()
    assert guard.cumulative_response_ready
    assert len(guard._response_error) >= 2
    assert guard.observation_travel_m < .008
    assert "projected_separation_conditional_empirical_evidence" in reasons
    assert not guard.improving
    # Boundary injection changes only the ledger, never the image fit/validation.
    guard.observation_travel_m = .020
    result = assess(guard, monitor, [-.002, 0, 0], translation_response=robot)
    assert result["accepted"] and result["reason"] == "projected_separation_conditional_empirical_evidence"
    assert result["cap_m"] <= .005 and not result["diagnostics"]["clearance_certified"]


def test_validated_model_does_not_admit_image_approach_or_observed_worsening():
    guard, monitor, state, robot, _ = learn_separation()
    guard.observation_travel_m = .020
    # Known analytic robot response in the synthetic plant; the RGB predictor
    # above was actually learned and evaluated on held-out observed frames.
    class PlantResponse:
        def predict(self, command, state):
            return {"supported": True, "responsive": True,
                    "mean_delta_m": (np.asarray(command)*.5).tolist()}
    assert not assess(guard, monitor, [.002, 0, 0], translation_response=PlantResponse())["accepted"]
    assert not assess(guard, monitor, [-.002, 0, 0], nominal_away=False,
                      translation_response=robot)["accepted"]
    guard.worsening.add("neighbor")
    assert assess(guard, monitor, [-.002, 0, 0], translation_response=robot)["reason"] == "translation_observed_neighbor_approach"


@pytest.mark.parametrize("failure", ["geometry", "track", "neighbor", "domain"])
def test_finite_recovery_never_overrides_existing_refusals(failure):
    guard, monitor, state = plant("separate")
    kwargs = {}
    if failure == "geometry": kwargs["nominal_clear"] = False
    if failure == "track": monitor.tracks["moving"].reliable = False
    if failure == "neighbor": guard.neighbors[0]["track"].reliable = False
    if failure == "domain": guard.domain_invalid = "translation_tool_configuration_changed"
    assert not assess(guard, monitor, [0, 0, .002], **kwargs)["accepted"]
    assert not guard.experiments


def test_saved_v13_upward_recovery_is_model_clear_away_but_downward_is_not():
    from gpt_field.visual_field_v13 import MetricFieldExecutorV13
    from gpt_field.contact_geometry_v13 import TargetContactGeometry
    from gpt_field.visual_field_v11 import _distance_to_box
    path = Path(__file__).parents[1]/"runs/visual_field_v13_zero_shot_002/trials/bowl_original_zero_hybrid_v13/trace.jsonl"
    if not path.exists(): pytest.skip("local archived RGB experiment unavailable")
    geometry = TargetContactGeometry()
    observation = None
    for line in path.open():
        row = json.loads(line)
        if row["event"] in ("reset", "executed"):
            observation = row["observation"]
        elif row["event"] == "action" and row["diagnostics"].get("refreshed"):
            prediction = row["diagnostics"]["prediction"]
            intent = prediction.get("field_intent")
            state = observation["observation/state"]
            geometry.configure(state, prediction["target_geometry"], target_id=prediction["target_id"],
                               tool_radius_m=.055 if intent is None else intent["tool_radius_m"])
            if row["step"] == 22:
                local = MetricFieldExecutorV13();local.contact_geometry = geometry
                local.configure(state, intent, scene_assessment=prediction["motion_context"])
                point = np.asarray(state[:3])
                parts = next(r for r in local.region_geometry if r["id"] == "cabinet")["tool_core_parts"]
                distance = lambda p: min(float(_distance_to_box(p, b["lower_xyz"], b["upper_xyz"])) for b in parts)
                up, down = point+[0, 0, .008], point-[0, 0, .008]
                assert local.checked.segment_valid(point, up)
                assert local.checked.segment_valid(point, down)
                assert distance(up)-distance(point) == pytest.approx(.0053181828362)
                assert distance(down)-distance(point) == pytest.approx(-.00476967709618)
                return
    pytest.fail("archive recovery action22 absent")
