"""V3-only low-overhead evaluator adapters; benchmark truth stays in the evaluator.

The private evaluator's live status function is adapted structurally. Its action,
feedback, collision, success and final artifact paths are left intact.
"""
from __future__ import annotations

import ast
import inspect
import math
import time

PERFORMANCE_VERSION = "agent-speedup-v3-performance-6"
LIVE_SNAPSHOT_INTERVAL_S = 1.0
COMPLETION_TARGET_S = 300.0


def performance_settings():
    return {
        "version": PERFORMANCE_VERSION,
        "live_snapshot_interval_s": LIVE_SNAPSHOT_INTERVAL_S,
        "live_policy_stats": "compact_current_state; full terminal agent_stats retained",
        "action_feedback_contact_trace": "complete, unthrottled",
        "video": "existing public RGB; 640x384; one frame per executed action; synchronous encoder drain",
        "completion_target_s": COMPLETION_TARGET_S,
        "target_scope": "preparation plus reviewed execution including recording and finalization; approval wait excluded",
        "target_is_stop_condition": False,
    }


class LiveSnapshotGate:
    def __init__(self, *, clock=time.monotonic, interval_s=LIVE_SNAPSHOT_INTERVAL_S):
        self.clock, self.interval_s = clock, float(interval_s)
        self.next_snapshot = 0.0
        self.published = self.skipped = 0

    def due(self, stage):
        now = self.clock()
        # Terminal/initial snapshots remain immediate even within the interval.
        if stage in ("deciding", "executing", "observed") and now < self.next_snapshot:
            self.skipped += 1
            return False
        self.next_snapshot = now + self.interval_s
        self.published += 1
        return True

    @staticmethod
    def policy_stats(policy):
        fast = getattr(policy, "fast_stats", None)
        if callable(fast):
            return fast()
        # Never fall back to a growing full history on the live status path.
        return {"step": getattr(policy, "step_index", None), "snapshot": "current_state_only",
                "full_stats": "agent_stats.json at finalization"}


class RecordingTimings:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.seconds = {"trace_append_s": 0.0, "artifact_json_write_s": 0.0}
        self.counts = {"trace_appends": 0, "artifact_json_writes": 0}

    def wrap(self, function, seconds_key, count_key):
        def measured(*args, **kwargs):
            started = self.clock()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[seconds_key] += self.clock() - started
                self.counts[count_key] += 1
        return measured

    def snapshot(self):
        return {**self.seconds, **self.counts,
                "scope": "private base evaluator, includes startup/finalization; nested in wall/control timers"}


def adapt_private_evaluator(module, *, clock=time.monotonic):
    """Adapt only the supplied fresh module, without modifying shared v2 modules.

    Fail closed if the reference state function changes shape. Compiling only
    execute preserves its original module globals and all evaluator semantics.
    """
    if getattr(module, "_v3_performance_adapted", False):
        return
    tree = ast.parse(inspect.getsource(module.execute))
    execute = tree.body[0]
    states = [node for node in execute.body if isinstance(node, ast.FunctionDef) and node.name == "state"]
    if len(states) != 1:
        raise RuntimeError("v3 performance adapter requires one evaluator state function")
    state = states[0]
    assignments = [i for i, node in enumerate(state.body) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "stage" for target in node.targets)]
    references = [node for node in ast.walk(state) if isinstance(node, ast.Attribute)
                  and node.attr == "agent_stats" and isinstance(node.value, ast.Name) and node.value.id == "policy"]
    if len(assignments) != 1 or len(references) != 1:
        raise RuntimeError("v3 performance adapter no longer matches live state semantics")
    state.body[assignments[0]+1:assignments[0]+1] = ast.parse(
        "if not _v3_live_snapshot_gate.due(new_stage):\n    return\n").body
    # The small snapshot is cheap at every published stage. Avoid a status file
    # indefinitely saying 'read at observed stage' when the throttle happens to
    # admit deciding/executing rather than observed.
    for node in state.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "light"
                                                for target in node.targets):
            node.value = ast.Constant(value=False)

    class CompactStats(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if (node.attr == "agent_stats" and isinstance(node.value, ast.Name)
                    and node.value.id == "policy"):
                return ast.copy_location(ast.parse(
                    "lambda: _v3_live_snapshot_gate.policy_stats(policy)", mode="eval").body, node)
            return self.generic_visit(node)

    CompactStats().visit(state)
    ast.fix_missing_locations(tree)
    gate, timings = LiveSnapshotGate(clock=clock), RecordingTimings(clock)
    module._v3_live_snapshot_gate = gate
    module._v3_recording_timings = timings
    module._append = timings.wrap(module._append, "trace_append_s", "trace_appends")
    module.write_json = timings.wrap(module.write_json, "artifact_json_write_s", "artifact_json_writes")
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    module._v3_performance_adapted = True


def completion_target(summary, measured_execution_s):
    """A reporting target, never a timeout or an alternative success criterion."""
    preparation = summary.get("user_check", {}).get("preparation_elapsed_s")
    if not isinstance(preparation, (float, int)) or not math.isfinite(preparation) or preparation < 0:
        total = None
    else:
        total = float(preparation) + float(measured_execution_s)
    success = summary.get("libero_success") is True and summary.get("task_success") is True
    return {"target_s": COMPLETION_TARGET_S, "preparation_elapsed_s": preparation,
            "reviewed_execution_elapsed_s": float(measured_execution_s),
            "total_service_elapsed_s": total,
            "achieved": bool(success and total is not None and total <= COMPLETION_TARGET_S),
            "raw_libero_success_required": True, "hard_stop": False,
            "includes": "preparation, execution, video close/drain, artifact normalization and runtime freeze",
            "excludes": "approval/operator wait; this final timing report's own serialization"}
