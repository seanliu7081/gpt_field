"""Placement control changes scoped to the user-check variant.

The legacy primitive treats four capped lateral corrections as unreachable even
when their residual error is small relative to the measured destination's
tolerance. Here that failure needs sustained poor correction outside half of
that tolerance. The hard drift bound, actions, contact and hold checks remain
the legacy controller's. These are disclosed design defaults, not a claim of
physical validation or evaluator-derived placement tolerances.
"""

from __future__ import annotations

from types import FunctionType, SimpleNamespace

import numpy as np

from . import agent_speedup_v2_primitives as primitives
from . import agent_speedup_v2_relations as relations
from .agent_speedup_v2_types import th_value, thresholds_dicts
from .agent_speedup_v2_user_check_constraints import (
    ScopedKeepOutObserver, preserved_preemption_info,
)


CONTROL_VERSION = "agent-speedup-v2-user-check-control-3"
LATERAL_STALL_STEPS = 12
LATERAL_TOLERANCE_FRACTION = 0.5


def control_settings() -> dict:
    """JSON-safe settings for the review binding, run manifest and diagnostics."""
    return {
        "version": CONTROL_VERSION,
        "scope": "scoped constraints, contact semantics, causal feedback and placement handoff",
        "constraint_scope": "current intent plus global claims; future steps are inactive",
        "contact_semantics": "explicit constraints prevail; support is pair and phase scoped",
        "checkpoints": ["grasp_confirmed", "final_alignment"],
        "destination_readiness": "empty-hand registration, measured top, planner region cells, support cue",
        "lateral_stall_steps": LATERAL_STALL_STEPS,
        "lateral_tolerance_fraction": LATERAL_TOLERANCE_FRACTION,
        "lateral_stall_error_m": "max(lateral_cap_m, lateral_tolerance_fraction * drift_abort_m)",
        "lateral_correction_cap_m": primitives.P15_LATERAL_CAP_M,
        "insufficient_progress_fraction": primitives.M5_SATURATED_REALIZED,
        "hard_drift_bound": "unchanged; measured destination's drift_abort_m",
        "threshold_basis": "design defaults; no evaluator or simulator truth inputs",
    }


class RelaxedGuardedLower(primitives.GuardedLower):
    """Allow in-tolerance transients without disabling bounded local recovery.

An unreachable event needs 12 consecutive steps whose before AND after error
exceed max(the 3 mm correction cap, half the destination drift tolerance), with
improvement <= 5% of that correction cap per step. Returning into the inner
tolerance or making adequate progress resets that count. Escaping the original
drift bound still fails immediately, even in this longer observation window.
"""

    def __init__(self, *args, **kwargs):
        # Base construction can finish immediately and request our evidence.
        self.tolerance_saturated_steps = 0
        super().__init__(*args, **kwargs)

    @property
    def lateral_stall_error_m(self) -> float:
        return max(self.lateral_cap_m, LATERAL_TOLERANCE_FRACTION * self.drift_abort_m)

    def _pre_event(self, before, after):
        event = super()._pre_event(before, after)
        error_before = float(np.linalg.norm(self.lateral_offset(before.position)))
        error_after = float(np.linalg.norm(self.lateral_offset(after.position)))
        outside_inner_tolerance = min(error_before, error_after) > self.lateral_stall_error_m
        insufficient_progress = (
            error_before - error_after <= primitives.M5_SATURATED_REALIZED * self.lateral_cap_m
        )
        self.tolerance_saturated_steps = (
            self.tolerance_saturated_steps + 1 if outside_inner_tolerance and insufficient_progress else 0
        )
        if event in ("hold_lost", "lateral_drift"):
            return event
        if self.tolerance_saturated_steps >= LATERAL_STALL_STEPS:
            return "lateral_unreachable"
        if event != "lateral_unreachable":
            return event
        # The old saturation event precedes G8 in the base method. Preserve a
        # same-step contact cue when suppressing that premature failure.
        return "gap_rise" if self._lower_steps_mm[-1] >= self.g8.value else None

    def _extra_evidence(self):
        evidence = super()._extra_evidence()
        thresholds = [t for t in evidence["thresholds"] if t["name"] != "M5_saturated_steps"]
        thresholds.extend(thresholds_dicts([
            th_value("M5_saturated_steps", LATERAL_STALL_STEPS, "DD",
                     "user-check design: longer consecutive lateral-correction observation window"),
            th_value("M5_stall_error_m", self.lateral_stall_error_m, "ONLINE",
                     "max(lateral correction cap, 0.5 * destination-derived drift_abort_m)"),
        ]))
        evidence.update(
            control_version=CONTROL_VERSION,
            lateral_stall_error_m=self.lateral_stall_error_m,
            saturated_steps=self.tolerance_saturated_steps,
            cap_saturated_steps=self.saturated_steps,
            thresholds=thresholds,
        )
        return evidence


def _rebind(function, **overrides):
    """Give a function private globals while retaining its code and closure."""
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = dict(function.__annotations__)
    bound.__dict__.update(function.__dict__)
    bound.__doc__ = function.__doc__
    bound.__qualname__ = function.__qualname__
    bound.__module__ = function.__module__
    return bound


def _clone_function(function, namespace):
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = dict(function.__annotations__)
    bound.__dict__.update(function.__dict__)
    bound.__doc__, bound.__qualname__, bound.__module__ = function.__doc__, function.__qualname__, function.__module__
    return bound


def refresh_relation_layer(layer):
    """Publish private hooks after installers replace helper functions.

    Every native helper resolves the same private namespace, so a nested grasp
    or lower call uses the installed hook too. No baseline globals are changed.
    """
    namespace = layer.USER_CHECK_NAMESPACE
    for name, value in namespace.items():
        if not name.startswith("__"):
            setattr(layer, name, value)
    controllers = {}
    for name in relations.CONTROLLERS:
        controller = namespace.get(name, namespace["not_implemented"])
        controllers[name] = namespace["_guarded"](controller) if name in {"hold", "turn", "place_on"} else controller
    layer.CONTROLLERS = controllers
    return layer


def _install_contact_monitor(ctx, namespace):
    """Keep the active intent's contact guard alive through local recovery."""
    current = ctx.monitor.observer("keepout") if "keepout" in ctx.monitor.keys() else None
    if isinstance(current, ScopedKeepOutObserver):
        return
    if not ctx.scene.keepouts:
        return
    monitor = ScopedKeepOutObserver(ctx=ctx, keepouts=list(ctx.scene.keepouts.values()),
                                    robot=ctx.robot, held=ctx.holding)
    namespace["_bus_add"](ctx, "keepout", monitor, preempt=True, scope="episode")


def _install_control_hooks(namespace):
    native_bus = namespace["_register_held_bus"]
    def register_held_bus(ctx, **kwargs):
        keys = native_bus(ctx, **kwargs)
        _install_contact_monitor(ctx, namespace)
        return keys
    namespace["_register_held_bus"] = register_held_bus

    native_grasp = namespace["_grasp_with_fallback"]
    def grasp_with_fallback(ctx, step, *, purpose):
        outcome = yield from native_grasp(ctx, step, purpose=purpose)
        callback = ctx.memory.get("user_check_checkpoint")
        if callback is not None and (outcome is None or outcome.status == "succeeded"):
            checkpoint = yield from callback("grasp_confirmed", step, {
                "holding": getattr(ctx.holding, "pid", None),
                "grip": getattr(ctx.holding, "grip_pid", None),
                "native_verification": None if outcome is None else outcome.reason,
                "verification_flags": [] if outcome is None else (outcome.evidence or {}).get("flags", [])})
            if not checkpoint.get("resume", False):
                return namespace["_outcome"](ctx, "held", "failed", "precondition_unmet",
                    pids=namespace["_pids"](step.obj, list(step.grip), step.ref),
                    evidence={"checkpoint": checkpoint, "cause": "checkpoint_repair_requested",
                              "actual_grasp": "verified" if ctx.holding is not None else "unknown"})
        return outcome
    namespace["_grasp_with_fallback"] = grasp_with_fallback

    native_failure = namespace["_grasp_failure"]
    def grasp_failure(outcome):
        thresholds, extra = native_failure(outcome)
        extra["grasp_failure"] = {"reason": outcome.reason, "status": outcome.status,
                                  "evidence": outcome.evidence}
        return thresholds, extra
    namespace["_grasp_failure"] = grasp_failure

    native_lower = namespace["_lower"]
    def lower(ctx, lowering, aim):
        callback = ctx.memory.get("user_check_checkpoint")
        step = ctx.memory.get("user_check_step")
        if callback is not None and step is not None:
            checkpoint = yield from callback("final_alignment", step, {
                "target_grip_xy_m": aim.get("target_grip_xy"), "target_object_xy_m": aim.get("target_obj_xy"),
                "target_grip_z_m": lowering.get("bound_grip_z"), "tolerance_m": aim.get("tol_m"),
                "lowering": {k: v for k, v in lowering.items() if k != "thresholds"}})
            if not checkpoint.get("resume", False):
                return {"reason": "precondition_unmet", "event": None, "g_event": None,
                        "lateral_error_m": None, "phases": [], "checkpoint": checkpoint,
                        "cause": "checkpoint_repair_requested"}
            if ctx.holding is None:
                return {"reason": "hold_lost", "event": None, "g_event": None,
                        "lateral_error_m": None, "phases": [], "checkpoint": checkpoint}
            if checkpoint.get("changed"):
                destination = ctx.scene.parts[step.ref]
                held = ctx.holding
                fresh = namespace["planner"].aim_target(destination, held, ctx.scene.hf,
                    namespace["_keepouts"](ctx, skip=(step.obj, getattr(held, "grip_pid", None))),
                    namespace["_pose"](ctx))
                if not fresh.get("ok"):
                    return {"reason": namespace["_reason_code"](fresh.get("reason"), "lateral_unreachable"),
                            "event": None, "g_event": None, "lateral_error_m": None, "phases": [],
                            "checkpoint": checkpoint, "recomputed_aim": namespace["jsonable"](fresh)}
                aim.clear()
                aim.update(fresh)
                lowering.clear()
                lowering.update(namespace["planner"].lowering_plan(destination, held, aim,
                    sigma_touch_m=namespace["_sigma_touch"]()))
        # Relay the existing generator protocol unchanged, retaining the
        # primitive evidence that legacy _lower's short summary discarded.
        controller = native_lower(ctx, lowering, aim)
        reply = None
        primitive_reports = []
        while True:
            try:
                request = controller.send(reply)
            except StopIteration as stopped:
                result = stopped.value
                result["primitive_feedback"] = primitive_reports
                return result
            reply = yield request
            if callable(request) and isinstance(reply, tuple) and isinstance(reply[0], dict):
                primitive_reports.append(namespace["jsonable"](reply[0]))
    namespace["_lower"] = lower

    # The semantic contact phase lives for the generator's actual execution,
    # including every yielded primitive and the resulting observer update.
    for name, phase in {"grasp": "grasp", "held_move": "transport", "_lower": "lower",
                        "_release_grip": "release"}.items():
        native = namespace[name]
        def phase_wrapper(ctx, *args, _native=native, _phase=phase, **kwargs):
            previous = ctx.memory.get("user_check_contact_phase")
            ctx.memory["user_check_contact_phase"] = _phase
            try:
                return (yield from _native(ctx, *args, **kwargs))
            finally:
                if previous is None:
                    ctx.memory.pop("user_check_contact_phase", None)
                else:
                    ctx.memory["user_check_contact_phase"] = previous
        namespace[name] = phase_wrapper

    # Install an empty-hand guard too. The native guard starts only after a
    # grasp and otherwise cannot report hand contacts during local recovery.
    native_guarded = namespace["_guarded"]
    def guarded(controller):
        native = native_guarded(controller)
        def run(ctx, step):
            ctx.memory["user_check_step"] = step
            _install_contact_monitor(ctx, namespace)
            return (yield from native(ctx, step))
        run.__name__ = getattr(controller, "__name__", "controller")
        return run
    namespace["_guarded"] = guarded
    native_release = namespace["release"]
    def release(ctx, step):
        if step is not None:
            ctx.memory["user_check_step"] = step
        _install_contact_monitor(ctx, namespace)
        return (yield from native_release(ctx, step))
    namespace["release"] = release


def build_relation_layer(base=None):
    """Create one isolated relation namespace for all user-check extensions."""
    if base is not None and base is not relations:
        return base
    from .agent_speedup_v2_user_check_skills import destination_check, destination_need

    primitive_view = SimpleNamespace(**vars(primitives))
    primitive_view.GuardedLower = RelaxedGuardedLower
    namespace = dict(vars(relations))
    for name, value in vars(relations).items():
        if isinstance(value, FunctionType) and value.__globals__ is vars(relations):
            namespace[name] = _clone_function(value, namespace)
    namespace.update(primitives=primitive_view, _destination_check=destination_check,
                     _destination_ok=destination_need, _preemption_info=preserved_preemption_info)
    _install_control_hooks(namespace)
    layer = SimpleNamespace(USER_CHECK_NAMESPACE=namespace)
    layer.USER_CHECK_CONTROL_VERSION = CONTROL_VERSION
    return refresh_relation_layer(layer)
