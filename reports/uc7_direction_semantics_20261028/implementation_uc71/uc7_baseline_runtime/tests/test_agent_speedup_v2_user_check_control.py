"""Offline controller evidence, without model calls or a simulator rollout."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import agent_speedup_v2_fakes as fakes
from gpt_field import agent_speedup_v2_primitives as legacy
from gpt_field import agent_speedup_v2_relations as relations
from gpt_field.agent_speedup_v2_user_check_control import (
    CONTROL_VERSION, LATERAL_STALL_STEPS, RelaxedGuardedLower, build_relation_layer, control_settings,
)
from gpt_field.contact_skills import StepView
from gpt_field.observation import PolicyObservation


# Archived t06 action boundaries 1762..1771, copied from trace.jsonl state8.
# This fixture checks the abort decision on recorded feedback. It does not
# simulate the unobserved continuation or establish physical task success.
T06_STATES = (
    (0.14889914926290893, -0.0018013768356269868, 0.6584094832762729, 2.4870172409871194, 1.9127453572360453, -0.002827805345215172, 0.0028385676826894334, -0.002862586801052092),
    (0.1505872215967687, -0.001083633213801804, 0.6569171924749713, 2.489130900969205, 1.9142530468513734, -0.008431295876050503, 0.0028390310239995078, -0.002862123716047336),
    (0.15215109605668922, -0.0006029167634613235, 0.653103907938704, -2.4895685892786648, -1.9145348223240035, 0.013616540422610754, 0.002839803763049602, -0.0028613447101984873),
    (0.15342066949567004, -0.00028690505624018553, 0.6488820199716837, -2.487928248469089, -1.9132983014576397, 0.017854611823732373, 0.00284106027438213, -0.0028600896047542047),
    (0.15436790124010552, -9.478156428262085e-05, 0.6446076658514575, -2.4866278461057227, -1.9123875991384367, 0.021040963710661446, 0.0028424692670478246, -0.0028586829578872573),
    (0.15500928277676487, 7.916175366430589e-06, 0.6403130589191774, -2.48569393828462, -1.9118064898447185, 0.023130254146446583, 0.0028439291463875143, -0.0028572242856115234),
    (0.15544147784323575, 4.6296933702006586e-05, 0.6359772088135174, -2.485212428540848, -1.9115995932745444, 0.023955137533457332, 0.0028451603420035702, -0.0028559934974454132),
    (0.15571960976364357, 4.187558789792736e-05, 0.6315509044829724, -2.4851505487490186, -1.9117264630247481, 0.023647292335458935, 0.0028463576044320377, -0.0028547961165852113),
    (0.15583742108760595, 1.3038625385218105e-05, 0.6270037401418951, -2.4854106860720635, -1.9120993302295037, 0.022502759997246886, 0.002847511896801379, -0.0028536417109027005),
    (0.15579368880309838, -2.7885841301527475e-05, 0.6223306843990494, -2.4859077556684013, -1.9126401533932744, 0.02078064436238964, 0.002848579565047666, -0.0028525740042010415),
)


def view(state=None, *, x=0.0, z=1.0, gap_mm=20.0):
    if state is None:
        state = (x, 0.0, z, math.pi, 0.0, 0.0, gap_mm / 2000, -gap_mm / 2000)
    blank = fakes._blank_url()
    return StepView(PolicyObservation(blank, blank, tuple(state), "offline lowering guard"))


def lower(start=None, *, cls=RelaxedGuardedLower, drift=0.010, **kwargs):
    return cls(start or view(), phase=kwargs.pop("phase", "slow"), travel_m=kwargs.pop("travel_m", 0.2),
               aim_grip_xy=(0.0, 0.0), drift_abort_m=drift, hold_floor_mm=5.0, **kwargs)


def advance(primitive, before, *, x=None, gap_mm=None, dz=None):
    action = primitive.next_action(before)
    state = np.asarray(before.state).copy()
    state[2] += action[2] * 0.05 * 0.26 if dz is None else dz
    if x is not None:
        state[0] = x
    if gap_mm is not None:
        state[6:8] = [gap_mm / 2000, -gap_mm / 2000]
    after = view(state)
    primitive.observe_transition(before, action, after)
    return after


def test_recorded_t06_transient_old_aborts_new_keeps_identical_commands():
    initial = view(T06_STATES[0])
    settings = dict(phase="fast", travel_m=initial.position[2] - 0.5775696935836958,
                    aim_grip_xy=(0.15145708778387554, 0.00013079940958639),
                    drift_abort_m=0.010202595269097848, hold_floor_mm=1.5)
    old, new = legacy.GuardedLower(initial, **settings), RelaxedGuardedLower(initial, **settings)
    for before_state, after_state in zip(T06_STATES, T06_STATES[1:]):
        before, after = view(before_state), view(after_state)
        old_action, new_action = old.next_action(before), new.next_action(before)
        assert old_action == new_action
        old.observe_transition(before, old_action, after)
        new.observe_transition(before, new_action, after)
    assert (old.done, old.reason, old.actions_executed) == (True, "lateral_unreachable", 9)
    assert not new.done and new.actions_executed == 9
    assert new.drift_m == pytest.approx(0.004339503359691515)
    assert new.lateral_stall_error_m == pytest.approx(0.005101297634548924)
    assert new.tolerance_saturated_steps == 0


@pytest.mark.parametrize("drift,soft", [(0.004, 0.003), (0.008, 0.004), (0.020, 0.010)])
def test_soft_guard_scales_with_measured_destination_tolerance(drift, soft):
    primitive = lower(drift=drift)
    assert primitive.lateral_stall_error_m == soft
    assert primitive.drift_abort_m == drift
    assert primitive.lateral_cap_m == legacy.P15_LATERAL_CAP_M


def test_true_divergence_still_stops_at_original_bound_before_window():
    before = view()
    primitive = lower(before)
    for i in range(1, 6):
        before = advance(primitive, before, x=0.0021 * i)
        assert primitive.done == (i == 5)
    assert primitive.reason == "lateral_drift"
    assert primitive.actions_executed < LATERAL_STALL_STEPS


def test_sustained_large_residual_still_fails_after_longer_window():
    before = view(x=0.007)
    primitive = lower(before)
    for i in range(1, LATERAL_STALL_STEPS + 1):
        before = advance(primitive, before)
        assert primitive.done == (i == LATERAL_STALL_STEPS)
    assert primitive.reason == "lateral_unreachable"
    evidence = primitive.outcome()["evidence"]
    assert evidence["saturated_steps"] == LATERAL_STALL_STEPS
    assert evidence["control_version"] == CONTROL_VERSION
    thresholds = {t["name"]: t["value"] for t in evidence["thresholds"]}
    assert thresholds["M5_saturated_steps"] == LATERAL_STALL_STEPS
    assert thresholds["M5_stall_error_m"] == 0.005


def test_small_stalled_error_allows_normal_travel_completion():
    before = view(x=0.004)
    primitive = lower(before, travel_m=0.030)
    while not primitive.done:
        before = advance(primitive, before)
    assert primitive.reason == "bound_reached"
    assert primitive.actions_executed > legacy.M5_SATURATED_STEPS


@pytest.mark.parametrize("recovered_x", [0.004, 0.0065])
def test_recovery_or_measurable_progress_resets_consecutive_counter(recovered_x):
    before = view(x=0.007)
    primitive = lower(before)
    for _ in range(7):
        before = advance(primitive, before)
    assert primitive.tolerance_saturated_steps == 7
    before = advance(primitive, before, x=recovered_x)
    assert primitive.tolerance_saturated_steps == 0 and not primitive.done


def test_hold_loss_has_priority_over_drift_and_saturation():
    before = view(x=0.007)
    primitive = lower(before)
    for _ in range(LATERAL_STALL_STEPS - 1):
        before = advance(primitive, before)
    advance(primitive, before, x=0.011, gap_mm=1.0)
    assert primitive.reason == "hold_lost"


@pytest.mark.parametrize("x,gap,reason", [(0.011, 20.0, "lateral_drift"), (0.0, 1.0, "hold_lost")])
def test_initial_hard_failure_is_unchanged(x, gap, reason):
    primitive = lower(view(x=x, gap_mm=gap))
    assert primitive.done and primitive.reason == reason
    assert primitive.actions_executed == 0


@pytest.mark.parametrize("phase,reason", [("slow", "gap_rise"), ("fast", "handover_gap_rise")])
def test_same_step_support_cue_is_not_hidden_by_suppressed_legacy_failure(phase, reason):
    before = view(x=0.004)
    primitive = lower(before, phase=phase)
    for _ in range(legacy.M5_SATURATED_STEPS - 1):
        before = advance(primitive, before)
    advance(primitive, before, gap_mm=22.0)
    assert primitive.done and primitive.reason == reason


def test_vertical_contact_stall_still_hands_over():
    before = view(x=0.004)
    primitive = lower(before, phase="fast")
    for _ in range(3):
        before = advance(primitive, before, dz=0.0)
    assert primitive.reason == "handover_contact_stall"


def test_primitive_budget_still_applies():
    before = view(x=0.004)
    primitive = lower(before)
    primitive.max_steps = 3
    for _ in range(3):
        before = advance(primitive, before)
    assert primitive.reason == "budget"


def test_relation_layer_is_scoped_and_preserves_other_controllers():
    original_lower, original_place = relations._lower, relations.place_on
    original_controllers = dict(relations.CONTROLLERS)
    layer = build_relation_layer()
    other_layer = build_relation_layer()
    assert layer._lower is not other_layer._lower
    assert relations._lower is original_lower and relations.place_on is original_place
    assert relations.CONTROLLERS == original_controllers
    assert legacy.GuardedLower is not RelaxedGuardedLower
    assert relations._lower.__globals__["primitives"].GuardedLower is legacy.GuardedLower
    assert layer.place_on.__globals__["_lower"] is layer._lower
    assert layer.CONTROLLERS["place_on"].__wrapped__ is layer.place_on
    for key in original_controllers.keys() - {"place_on"}:
        assert layer.CONTROLLERS[key] is original_controllers[key]
    assert layer._lower.__globals__["primitives"].Touchdown is legacy.Touchdown
    stand_in = SimpleNamespace(CONTROLLERS={})
    assert build_relation_layer(stand_in) is stand_in


def test_variant_relation_uses_relaxed_primitive_for_fast_and_slow_phases():
    layer = build_relation_layer()
    layer._lower.__globals__["_g6"] = lambda ctx: SimpleNamespace(value=5.0)
    ctx = SimpleNamespace(view=view())
    gen = layer._lower(ctx, {"fast_until_grip_z": 0.95, "bound_grip_z": 0.90, "drift_abort_m": 0.010},
                       {"target_grip_xy": (0.0, 0.0)})
    fast = next(gen)(ctx.view)
    assert type(fast) is RelaxedGuardedLower and fast.phase == "fast"
    ctx.view = view(z=0.95)
    slow = gen.send(({"status": "succeeded", "reason": "travel_reached", "actions": 12}, fast))(ctx.view)
    assert type(slow) is RelaxedGuardedLower and slow.phase == "slow"
    gen.close()
    assert control_settings()["version"] == CONTROL_VERSION
