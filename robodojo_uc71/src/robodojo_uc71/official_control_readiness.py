"""Static evidence checks for official EEF control; never launch or certify a run.

Installation/import/IK-only checks do not qualify native physical execution.
A separate current-host diagnostic must supply reviewed motion evidence. Legacy
Jacobian diagnostics remain geometry evidence, not official-controller evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .approval import VerifiedFileHashCache


CHECKS = frozenset({"tcp_to_link6", "both_arm_native_ik", "inactive_cartesian_hold",
    "native_gripper_mapping", "native_ik_failure_semantics", "native_action_accounting",
    "measured_arrival_and_gripper_timing", "current_host_reset", "gpu_isolation_and_cleanup"})
_files = VerifiedFileHashCache()


def verify_official_readiness(binding, *, geometry, reset_reference, parameters):
    """Require backend-specific measured evidence, not just a new success flag."""
    from .official_control_setup import verify_setup_record
    if not isinstance(binding, dict) or set(binding) != {"setup", "qualification"}:
        raise ValueError("official_setup_and_native_qualification_required")
    cache = _files

    def read(entry):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("bound_official_control_evidence_required")
        cache.verify(entry["path"], entry["sha256"])
        value = json.loads(Path(entry["path"]).read_bytes())
        cache.verify(entry["path"], entry["sha256"])
        return value

    read(binding["setup"])
    setup = verify_setup_record(binding["setup"]["path"])
    proof = read(binding["qualification"])
    if (proof.get("schema") != "robodojo-official-control-qualification-1"
            or proof.get("control_backend") != "official_ee" or proof.get("passed") is not True
            or proof.get("scope") != "current_host_native_ee_motion_and_gripper_diagnostic"
            or proof.get("model_calls") != 0 or proof.get("full_test_authorized") is not False
            or proof.get("setup") != binding["setup"] or proof.get("geometry") != geometry
            or proof.get("reset_reference") != reset_reference
            or proof.get("monitoring") != parameters["monitoring"]
            or proof.get("execution_limits") != parameters["execution_limits"]
            or set(proof.get("checks", {})) != CHECKS
            or any(value is not True for value in proof["checks"].values())):
        raise ValueError("fresh_measured_official_control_qualification_required")
    rows = proof.get("native_receipts")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("official_native_action_evidence_required")
    indices = []
    for row in rows:
        receipt = read(row)
        index = receipt.get("native_action_index")
        if (receipt.get("schema") != "robodojo-native-action-receipt-1"
                or receipt.get("control_backend") != "official_ee"
                or set(receipt.get("ik_status", {})) != {"left", "right"}
                or any(s not in ("Success", "Fail") for s in receipt["ik_status"].values())
                or type(index) is not int or index < 1
                or receipt.get("native_sim_step_calls") != 10
                or receipt.get("native_terminal_callback_calls") != 1
                or receipt.get("native_reward_callback_calls") != 1
                or receipt.get("physics_dt_s") != .004
                or not math.isclose(receipt["physics_time_after_s"]-receipt["physics_time_before_s"], .04,
                                    rel_tol=0., abs_tol=1e-7)):
            raise ValueError("official_native_action_evidence_invalid")
        indices.append(index)
    if indices != list(range(1, len(rows)+1)) or proof.get("native_actions") != len(rows):
        raise ValueError("official_native_action_sequence_invalid")
    required = [binding["setup"], geometry, reset_reference, *rows]
    sources = proof.get("source_files", [])
    for entry in sources:
        cache.verify(entry["path"], entry["sha256"])
    if not {(r["path"], r["sha256"]) for r in required} <= {(r["path"], r["sha256"]) for r in sources}:
        raise ValueError("official_qualification_source_closure_missing")
    return setup
