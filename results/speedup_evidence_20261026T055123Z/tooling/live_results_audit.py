#!/usr/bin/env python3
"""Collect terminal P5 evidence; never launch/score runs or read held-out truth.

The paired gate remains the responsibility of evaluation_summary.py. This tool
does not inspect model prompts/responses, infer missing terminal outcomes, or
turn release/settle diagnostic actions into autonomous task completion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def quality(trial: Path):
    truth = read(trial / "physical_truth.json", {})
    if not truth:
        return {"available": False}
    terminal = truth.get("terminal") or {}
    after = truth.get("after_release_settle") or {}

    def atoms(state):
        return [{"atom": row.get("atom"), "kind": row.get("kind"),
                 "raw": row.get("raw"), "strict": row.get("strict"),
                 "checks": row.get("checks"), "why_not": row.get("why_not")}
                for row in state.get("atoms", [])]

    return {
        "available": True,
        "truth_version": terminal.get("truth_version"),
        "diagnosis_only": truth.get("diagnosis_only"),
        "affects_completion": truth.get("affects_completion"),
        "terminal_strict_conjunction": terminal.get("strict_conjunction"),
        "terminal_raw_conjunction": terminal.get("raw_conjunction"),
        "terminal_atoms": atoms(terminal),
        "after_release_settle_strict_conjunction": after.get("strict_conjunction"),
        "after_release_settle_raw_conjunction": after.get("raw_conjunction"),
        "after_release_settle_atoms": atoms(after),
        "extra_diagnostic_actions": (after.get("settle") or {}).get("actions"),
        "extra_diagnostic_protocol": (after.get("settle") or {}).get("protocol"),
        "withdrawal_independently_verified": None,
        "note": "Terminal and after_release_settle are separate. The latter is assisted diagnostic evidence, never autonomous success. Withdrawal has no independent boolean certificate here.",
    }


def trace_counts(path: Path):
    """Count structural action provenance only; do not perform decision review."""
    if not path.exists():
        return {"available": False}
    events, execution, primitive = Counter(), Counter(), Counter()
    actions, model_refresh_actions, parse_errors = 0, 0, []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_errors.append(line_number)
                continue
            events[str(event.get("event"))] += 1
            if event.get("event") != "action":
                continue
            actions += 1
            diagnostics = event.get("diagnostics") or {}
            execution[str(diagnostics.get("execution_kind"))] += 1
            primitive[str(diagnostics.get("primitive"))] += 1
            model_refresh_actions += bool(diagnostics.get("api_calls_in_act"))
    return {"available": True, "events": dict(events), "actions": actions,
            "execution_kind": dict(execution), "primitive": dict(primitive),
            "model_refresh_actions": model_refresh_actions,
            "malformed_lines": parse_errors, "semantic_decision_windows_reviewed": 0}


def collect(root: Path):
    rows, pending, heldout = [], [], []
    for reservation_path in sorted((root / "reservations").glob("trial_*.json")):
        reservation = read(reservation_path)
        if reservation.get("event") != "reservation":
            continue
        identity = {key: reservation.get(key) for key in
                    ("reservation_id", "stage", "task", "episode_index", "seed", "arm", "retry_of")}
        if reservation.get("stage") != "P5":
            heldout.append(identity)
            continue  # Do not even open a held-out session's summary/truth.
        session = root / "evaluations" / reservation["reservation_id"]
        receipt_path = session / "evidence_result.json"
        if not receipt_path.exists():
            # Partial run_state/trace is never interpreted as a terminal result.
            pending.append({**identity, "terminal_receipt_present": False})
            continue
        receipt = read(receipt_path)
        trial = session / "trial"
        summary = read(trial / "summary.json", {})
        stats = read(trial / "agent_stats.json", {})
        quality_evidence = quality(trial)
        infrastructure = read(session / "infrastructure_annotation.json", {})
        model = summary.get("model_calls") or {}
        timings = summary.get("timing_components") or {}
        preparation = read(session / "service_clock.json", {})
        row = {
            **identity,
            "terminal_receipt_present": True,
            "terminal_summary_present": bool(summary),
            "status": receipt.get("status"),
            "raw_evaluator_summary_status": receipt.get("raw_evaluator_summary_status"),
            "raw_libero_success": receipt.get("raw_libero_success"),
            "time_to_raw_success_s": receipt.get("time_to_raw_success_s"),
            "service_elapsed_s": receipt.get("service_elapsed_s"),
            "human_review_wait_s": receipt.get("human_review_wait_s"),
            "automatic_review_compute_s": receipt.get("automatic_review_compute_s"),
            "clock": receipt.get("clock"),
            "service_clock_events": preparation.get("events"),
            "approved_by": receipt.get("approved_by"),
            "review_modified": receipt.get("review_modified"),
            "infrastructure_annotation": infrastructure or None,
            "steps": summary.get("steps", infrastructure.get("policy_actions")),
            "api_calls": summary.get("api_calls", infrastructure.get("model_transport_calls")),
            "model_calls": model,
            "preparation_request_files": len(list((session / "review" / "requests").glob("*.request.json"))),
            "execution_request_files": len(list((trial / "requests").glob("*.request.json"))),
            "action_source_counts": stats.get("counts"),
            "execution_owner": stats.get("execution_owner"),
            "trace_counts": trace_counts(trial / "trace.jsonl"),
            "autonomous_terminal_quality": quality_evidence,
            "operator_action_interventions": (receipt.get("autonomous_quality") or {}).get("operator_action_interventions"),
            "timing_components": timings,
            "control_elapsed_s": summary.get("control_elapsed_s"),
            "evaluator_elapsed_s": summary.get("elapsed_s"),
            "compact_video_timing": read(trial / "episode.timing.json"),
            "io_timing_caution": "video_render_s is evaluator timing; compact_video_timing fields overlap it and close_drain contains terminal render/write. Do not sum nested timers.",
            "collision_contact_point_counts": (summary.get("collision") or {}).get("counts"),
            "collision_count_caution": "Contact points across physics substeps are not independent collision episodes.",
            "error": summary.get("error"),
            "failure_reason": summary.get("failure_reason"),
            "cleanup_errors": summary.get("cleanup_errors"),
            "video_error": summary.get("video_error"),
            "evidence_sha256": {str(path.relative_to(root)): digest(path) for path in
                (receipt_path, trial / "summary.json", trial / "physical_truth.json", trial / "agent_stats.json") if path.exists()},
        }
        rows.append(row)
    return {
        "version": "terminal-evidence-audit-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "scope": "P5 terminal evidence only; held-out truth never opened; no gate calculation",
        "semantic_decision_windows_reviewed": 0,
        "attribution_limit": "Online A/B trajectories may differ through model sampling. Component replay establishes I/O savings; different task decisions/outcomes alone do not causally establish an I/O mechanism.",
        "terminal_receipts": rows,
        "pending_without_terminal_receipt": pending,
        "heldout_reservations_not_audited": heldout,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = collect(root)
    (root / "live_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = ["reservation_id", "stage", "task", "episode_index", "seed", "arm", "retry_of", "status", "raw_libero_success", "time_to_raw_success_s", "service_elapsed_s", "human_review_wait_s", "automatic_review_compute_s", "steps", "api_calls", "preparation_request_files", "execution_request_files", "terminal_strict_conjunction", "after_release_settle_strict_conjunction", "extra_diagnostic_actions", "model_pipeline_s", "simulator_step_s", "video_render_s", "run_state_s", "infrastructure_failure", "failure_reason", "error"]
    with (root / "live_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["terminal_receipts"]:
            flat = {key: row.get(key) for key in fields}
            flat.update({key: row["autonomous_terminal_quality"].get(key) for key in ["terminal_strict_conjunction", "after_release_settle_strict_conjunction", "extra_diagnostic_actions"]})
            flat.update({key: row["timing_components"].get(key) for key in ["model_pipeline_s", "simulator_step_s", "video_render_s", "run_state_s"]})
            flat["infrastructure_failure"] = (row["infrastructure_annotation"] or {}).get("infrastructure_failure", False)
            writer.writerow(flat)
    print(json.dumps({"terminal_receipts": len(result["terminal_receipts"]), "pending": len(result["pending_without_terminal_receipt"]), "heldout_not_audited": len(result["heldout_reservations_not_audited"])}))


if __name__ == "__main__":
    main()
