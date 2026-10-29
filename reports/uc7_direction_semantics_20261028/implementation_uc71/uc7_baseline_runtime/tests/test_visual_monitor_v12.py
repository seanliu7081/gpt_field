"""Causal cumulative motion evidence and persistent uncertainty regressions."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_monitor_v12 import RoleAwareVisualMonitorV12
from test_visual_monitor_v8 import views
from test_visual_field_v6_policy import observation, prediction


def test_subpixel_noisy_causal_predictions_are_evaluated_over_resolved_motion():
    monitor = RoleAwareVisualMonitorV12()
    # Per-frame jitter dominates each .08px response, while endpoints resolve
    # actual motion. These forecasts are frozen causal rows, not a refit.
    for i in range(24):
        monitor._prediction_window.append({"delta": np.array([.0004, 0., 0.]),
            "predicted": np.array([.08, 0.]),
            "observed": np.array([.08+(.15 if i%2 else -.15), 0.]), "noise": .12})
    evidence = monitor.cumulative_response_evidence()
    assert evidence["supported"] and evidence["held_out_transitions"] >= 3
    assert evidence["observed_displacement_px"][0] >= evidence["noise_threshold_px"]
    assert evidence["causal_predictions"] and not evidence["clearance_certified"]


def test_old_good_window_cannot_hide_recent_distinguishable_static_feature():
    monitor = RoleAwareVisualMonitorV12()
    for i in range(24):
        monitor._prediction_window.append({"delta": np.array([.001, 0., 0.]),
            "predicted": np.array([.3, 0.]), "observed": np.array([.3, 0.]), "noise": .08})
    for i in range(6):
        monitor._prediction_window.append({"delta": np.array([.001, 0., 0.]),
            "predicted": np.array([.3, 0.]), "observed": np.zeros(2), "noise": .08})
    evidence = monitor.cumulative_response_evidence()
    assert not evidence["supported"] and evidence["contradicted"]
    assert evidence["transitions"] < 8


def test_stationary_rgb_exhausts_finite_measured_budget_and_reseed_does_not_renew():
    monitor = RoleAwareVisualMonitorV12(); obs=observation(); refs=prediction()["visual_references"]
    monitor.configure(views(obs), refs, obs.state, intent_id=1)
    fixed=views(obs)
    for i in range(1, 32):
        state=np.array(obs.state);state[0]+=.001*i
        monitor.observe(fixed,state,[.1,0,0,0,0,0,0])
    assert not monitor.response_ready
    assert monitor.measured_control_limit()[0] == 0.
    used=monitor._active_reference["unvalidated_travel_m"]
    for mode in ("retain","seed"):
        new=copy.deepcopy(refs);new["external"]["update"]=mode
        if mode=="retain":new["external"]["seed"]=None
        monitor.configure(fixed,new,state,intent_id=10)
        assert monitor.measured_control_limit()[0] == 0.
        assert monitor._active_reference["unvalidated_travel_m"] == used


def test_real_rgb_slow_motion_produces_reuse_without_new_model_seed():
    monitor=RoleAwareVisualMonitorV12();obs=observation();refs=prediction()["visual_references"]
    monitor.configure(views(obs),refs,obs.state,intent_id=1)
    ready=[]
    for i in range(1,45):
        offset=np.array([i*.0005,0.,0.])
        # Slow actuator + low image gain: .5mm physical increments, .1px/frame.
        image_obs=observation(offset*.2)
        state=(*tuple(np.asarray(obs.state[:3])+offset),*obs.state[3:])
        monitor.observe(views(image_obs),state,[.05,0,0,0,0,0,0])
        ready.append(monitor.response_ready)
        assert monitor.measured_control_limit()[1] is None
    assert any(ready)
    assert monitor._active_reference["validated_transitions"] > 0


def test_grasp_evidence_is_not_created_by_cumulative_tool_prediction():
    monitor=RoleAwareVisualMonitorV12();obs=observation();refs=prediction()["visual_references"]
    monitor.configure(views(obs),refs,obs.state,intent_id=1)
    for i in range(1,12):
        after=observation((i*.001,0,0));monitor.observe(views(after),after.state,[.1,0,0,0,0,0,0])
    assert monitor.response_ready
    assert not monitor.attachment_evidence
    assert not monitor.summary()["cumulative_response"]["clearance_certified"]
