"""Decision context is bounded; audit evidence is never truncated on disk.

Keep the live blocker, its causal path, current state and reviewed goals. Motion
history contains measured changes, not copies of static controller instructions.
"""
from __future__ import annotations

import copy
import json

REQUEST_FIELD_BUDGET = 50000  # headroom below the native 60,000-character request limit


def size(value):
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def _bounded(value, budget=3000):
    if size(value) <= budget:
        return copy.deepcopy(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            candidate = _bounded(item, max(160, budget // max(1, len(value))))
            if size({**out, key: candidate}) <= budget - 80:
                out[key] = candidate
        out["detail_in_audit"] = True
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            candidate = _bounded(item, max(160, budget // max(1, min(len(value), 8))))
            if size(out + [candidate]) > budget - 80:
                break
            out.append(candidate)
        if len(out) < len(value):
            out.append({"more_in_audit": len(value) - len(out)})
        return out
    if isinstance(value, str):
        return value[:max(0, budget - 70)] + " [full text in audit]"
    return value


def causal_nodes(value):
    """Extract nested refusal measurements before applying a context budget.

    Shallow truncation otherwise retains wrapper dicts and loses the actual
    reason. Paths preserve which phase produced each measurement.
    """
    found = []
    def visit(item, path):
        if isinstance(item, dict):
            keys = {"cause", "hits", "unmet_conditions", "unmet", "failed", "violations",
                    "thresholds", "preemption", "blocker", "causal", "constraint_origin"}
            if keys.intersection(item) or (item.get("reason") and item.get("reason") not in
                    {"goal_reached", "registered", "hold_pass", "succeeded"}):
                measurements = {k: v for k, v in item.items() if not isinstance(v, (dict, list))}
                for k in ("hits", "unmet_conditions", "unmet", "thresholds", "values", "cause",
                          "causal", "preemption", "constraint_origin", "phase", "source"):
                    if k in item:
                        measurements[k] = item[k]
                found.append({"path": path, **_bounded(measurements, 1800)})
            for key, child in item.items():
                if isinstance(child, (dict, list)):
                    visit(child, path + "." + key)
        elif isinstance(item, (list, tuple)):
            for i, child in enumerate(item):
                if isinstance(child, (dict, list)):
                    visit(child, path + f"[{i}]")
    visit(value, "evidence")
    # Deepest reports usually contain the actual geometric check. Preserve an
    # explicit causal report first, then the terminal check and wrapper context.
    found.sort(key=lambda row: ("causal" not in row and "hits" not in row, -len(row["path"])))
    unique, seen = [], set()
    for row in found:
        signature = json.dumps({k:v for k,v in row.items() if k != "path"}, sort_keys=True)
        if signature not in seen:
            unique.append(row)
            seen.add(signature)
    return _bounded(unique[:12], 6500)


def feedback_view(value):
    if value is None:
        return None
    result = {k: copy.deepcopy(value[k]) for k in
              ("status", "skill", "reason", "executed", "policy_step") if k in value}
    result["unmet_conditions"] = _bounded(value.get("unmet_conditions", []), 3000)
    result["observed_effects"] = _bounded(value.get("observed_effects", []), 1500)
    evidence = value.get("evidence", {})
    result["causes"] = causal_nodes(evidence)
    result["controller"] = {k: copy.deepcopy(evidence[k]) for k in
                            ("controller_status", "controller_steps", "pids", "text") if k in evidence}
    result["thresholds"] = _bounded(evidence.get("thresholds", []), 1200)
    return result


def pose_view(value):
    return {k: copy.deepcopy(v) for k, v in (value or {}).items()
            if k not in {"limits", "controller", "instructions", "coordinate_contract", "orientation_contract",
                         "position_contract", "notes", "static_limits", "semantics", "axes", "units",
                         "position_reference", "orientation_reference", "measured_pose"}}


def motion_view(row):
    if not row:
        return None
    result = {k: copy.deepcopy(v) for k, v in row.items() if k not in {"before", "after", "evidence"}}
    before, after = row.get("before") or {}, row.get("after") or {}
    result["after"] = pose_view(after)
    # Keep explicit target and measured delta; static descriptions and two full
    # state snapshots do not accumulate with each short movement.
    result["evidence"] = _bounded(row.get("evidence", {}), 1400)
    return _bounded(result, 2600)


def framework_view(value):
    result = copy.deepcopy(value)
    result.pop("skills", None)  # contracts are in the fixed instruction, once
    task = result.get("task", {})
    task.pop("identity_rule", None)
    task.pop("completion_rule", None)
    for item in task.get("reviewed_intents", []):
        item.pop("key", None)  # duplicate serialized intent; IDs remain in audit
    result["last_feedback"] = feedback_view(value.get("last_feedback"))
    local = value.get("local_control", {})
    result["local_control"] = {
        "context": pose_view(local.get("context", {})),
        "last_feedback": motion_view(local.get("last_feedback")),
        "recent_segments": [motion_view(row) for row in local.get("recent_segments", [])[-3:-1]],
        **{k: _bounded(v, 2400) for k, v in local.items()
           if k not in {"context", "last_feedback", "recent_segments"}},
    }
    recovery = result.get("recovery", {})
    recovery["recent"] = [{"feedback": {k:v for k,v in row.get("feedback", {}).items()
                                       if k != "evidence"},
                            **{k:row[k] for k in ("n", "context", "consecutive_unchanged",
                                                 "recurring_condition_count") if k in row}}
                          for row in recovery.get("recent", [])[-3:]]
    recovery["recurring_conditions"] = _bounded(recovery.get("recurring_conditions", [])[-8:], 2200)
    return result


def bounded_fields(fields):
    """Reduce history first. Never delete or rewrite current goals/plan fields."""
    result = copy.deepcopy(fields)
    if size(result) <= REQUEST_FIELD_BUDGET:
        return result
    framework = result.get("execution_framework", {})
    local = framework.get("local_control", {})
    local.pop("recent_segments", None)
    recovery = framework.get("recovery", {})
    recovery["recent"] = recovery.get("recent", [])[-1:]
    recovery["recurring_conditions"] = _bounded(recovery.get("recurring_conditions", []), 900)
    if size(result) > REQUEST_FIELD_BUDGET:
        local["last_feedback"] = _bounded(local.get("last_feedback"), 1000)
        feedback = framework.get("last_feedback") or {}
        feedback["causes"] = _bounded(feedback.get("causes", []), 2500)
        framework["last_feedback"] = feedback
        state = framework.get("state", {})
        entities = state.get("entities", {})
        state["entities"] = {pid: {k:v for k,v in item.items() if k not in {"roles", "description"}}
                             for pid, item in entities.items()}
    if size(result) > REQUEST_FIELD_BUDGET:
        # The current native status block already carries every step. Historical
        # request input and old feedback can be summarized without changing it.
        history = result.get("step_history")
        if isinstance(history, dict):
            history["earlier"] = history.get("earlier", [])[-2:]
        framework["recovery"] = {"total_outcomes": recovery.get("total_outcomes"),
                                  "recurring_conditions": recovery.get("recurring_conditions", [])}
    if size(result) > REQUEST_FIELD_BUDGET:
        # Last resort removes only auxiliary detail, retaining complete current
        # and reviewed plans, current measured state and a causal current error.
        state = framework.get("state", {})
        state["evidence"] = _bounded(state.get("evidence", {}), 1800)
        for item in state.get("entities", {}).values():
            item.pop("region_evidence", None)
            item["geometry"] = _bounded(item.get("geometry", {}), 600)
        feedback = framework.get("last_feedback") or {}
        feedback["causes"] = _bounded(feedback.get("causes", []), 1400)
        feedback["thresholds"] = _bounded(feedback.get("thresholds", []), 600)
        feedback["unmet_conditions"] = _bounded(feedback.get("unmet_conditions", []), 1500)
        local["last_feedback"] = _bounded(local.get("last_feedback"), 800)
        framework["recovery"] = {"total_outcomes": recovery.get("total_outcomes"),
                                  "history_summary": "Older outcomes retained in execution audit"}
        result["context_compaction"] = "Current goals preserved; historical and auxiliary detail reduced"
    return result
