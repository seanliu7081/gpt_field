"""External-image direction map from the robot's own motion (agent_SpeedUp_v1 live run 001 fix)."""

import numpy as np
import pytest

from agent_speedup_fakes import FakeWristSim
from gpt_field.external_axes import fit_external_axes, hand_image_motion, image_shift_for


def frames(sim, offset):
    before = sim.external_upright()
    start = sim.state[:3].copy()
    sim.state[:3] = start + np.asarray(offset, float)
    after = sim.external_upright()
    return before, after, sim.state[:3] - start


def test_hand_image_motion_tracks_the_moving_hand_over_a_static_scene():
    sim = FakeWristSim(external_px_per_m=800.0)
    before, after, moved = frames(sim, (0.03, -0.02, 0.0))
    motion = hand_image_motion(before, after)
    # ORACLE fake camera: world +y -> image right, +x -> image down, 800 px/m
    assert motion["image_shift_px"] == pytest.approx([800 * moved[1], 800 * moved[0]], abs=3.0)
    still = sim.external_upright()
    assert hand_image_motion(still, still) is None


def test_fit_recovers_image_right_and_up_in_world_and_rejects_bad_probes():
    sim = FakeWristSim(external_px_per_m=800.0)
    samples = []
    for offset in ((0.04, 0.0, 0.0), (-0.04, 0.04, 0.0), (0.0, -0.04, 0.0)):
        before, after, moved = frames(sim, offset)
        samples.append({"world_displacement_m": moved.tolist(),
                        "image_shift_px": hand_image_motion(before, after)["image_shift_px"]})
    fit = fit_external_axes(samples)
    assert fit["available"] and fit["samples"] == 3 and max(fit["residual_deg"]) < 5
    assert fit["image_right_world"] == pytest.approx([0.0, 1.0, 0.0], abs=0.05)
    assert fit["image_up_world"] == pytest.approx([-1.0, 0.0, 0.0], abs=0.05)
    assert image_shift_for(fit, [0.0, 0.05, 0.0]) == pytest.approx([40.0, 0.0], abs=2.0)
    collinear = fit_external_axes([samples[0], {**samples[0], "world_displacement_m": [-0.04, 0.0, 0.0],
                                                "image_shift_px": [0.0, -32.0]}])
    assert not collinear["available"] and collinear["reason"] == "probe_directions_collinear"
    missing = fit_external_axes([{**s, "image_shift_px": None} for s in samples])
    assert not missing["available"] and missing["reason"] == "fewer_than_two_usable_probe_moves"
    inconsistent = fit_external_axes([samples[0], samples[1], {**samples[2], "image_shift_px": [32.0, 0.0]}])
    assert not inconsistent["available"] and inconsistent["reason"] == "inconsistent_probe_flow"
    assert image_shift_for(inconsistent, [0.0, 0.05, 0.0]) is None
