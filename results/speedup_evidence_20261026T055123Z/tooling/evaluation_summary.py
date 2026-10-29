"""Cumulative reservation ledger and paired reporting; no policy or truth imports."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def reserve(root, *, stage, task, episode_index, arm, retry_of=None, reason=None):
    """Debit before prepare; failures and unused reservations never refund budget."""
    root = Path(root).resolve()
    manifest = json.loads((root / "execution_manifest.json").read_text())
    if stage not in {"P4", "P5", "P6"} or arm not in {"A", "B"}:
        raise ValueError("invalid stage/arm")
    if stage == "P5" and manifest.get("stage_gates", {}).get("P2") != "passed":
        raise ValueError("P5 requires a recorded passing equivalence gate")
    if stage == "P6" and manifest.get("stage_gates", {}).get("P5") != "passed":
        raise ValueError("P6 requires a recorded passing P5 gate")
    if stage in {"P5", "P6"}:
        eligible = {(p["task"], p["episode_index"]) for p in manifest["splits"][stage]}
        if (task, episode_index) not in eligible:
            raise ValueError("condition not preregistered")
    ledger = root / "budget_ledger.jsonl"
    with ledger.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        entries = [json.loads(line) for line in stream if line.strip()]
        rows = [r for r in entries if r.get("event") == "reservation"]
        caps = manifest["budget_caps"]
        if retry_of is not None:
            originals = [r for r in rows if r["reservation_id"] == retry_of]
            if len(originals) != 1 or originals[0].get("retry_of"):
                raise ValueError("retry must identify one original reservation")
            original = originals[0]
            if any(original[k] != v for k, v in {"stage": stage, "task": task,
                                                "episode_index": episode_index, "arm": arm}.items()):
                raise ValueError("retry cannot change its original condition")
            if not reason or not reason.strip():
                raise ValueError("an evidenced infrastructure reason is required")
            if sum(bool(r.get("retry_of")) for r in rows) >= caps["infrastructure_retries_total"]:
                raise ValueError("infrastructure retry budget exhausted")
            if sum(r.get("retry_of") == retry_of for r in rows) >= caps["retries_per_original"]:
                raise ValueError("original retry allowance exhausted")
        else:
            key = "P4_rollouts" if stage == "P4" else stage + "_episodes"
            if sum(r["stage"] == stage and not r.get("retry_of") for r in rows) >= caps[key]:
                raise ValueError("stage budget exhausted")
            if stage != "P4" and any(r["stage"] == stage and r["task"] == task
                                      and r["episode_index"] == episode_index and r["arm"] == arm
                                      and not r.get("retry_of") for r in rows):
                raise ValueError("condition already reserved; no silent rerun")
        row = {"event": "reservation", "reservation_id": f"trial_{len(rows)+1:03d}",
               "experiment_id": manifest["experiment_id"], "stage": stage, "task": task,
               "episode_index": episode_index, "seed": 7, "arm": arm,
               "retry_of": retry_of, "infrastructure_reason": reason,
               "reserved_monotonic_s": time.monotonic(),
               "effective_service_budget_s": 180 if stage == "P4" else 900,
               "score_threshold_s": 300, "review_authorized": False,
               "note": "Budget reservation is not initial-plan approval; no refund on failure."}
        row["reservation_sha256"] = canonical_hash(row)
        destination = root / "reservations" / (row["reservation_id"] + ".json")
        destination.parent.mkdir(exist_ok=True)
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        with destination.open("x") as target:
            json.dump(row, target, ensure_ascii=False, indent=2)
        return destination


def score(raw, elapsed):
    return bool(raw is True and finite_seconds(elapsed) and elapsed <= 300)


def finite_seconds(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def measured_terminal(row):
    if row.get("status") in {None, "not_started", "awaiting_review", "running", "unknown"}:
        return False
    raw, elapsed, first = (row.get(k) for k in (
        "raw_libero_success", "service_elapsed_s", "time_to_raw_success_s"))
    return (type(raw) is bool and finite_seconds(elapsed)
            and (not raw or (finite_seconds(first) and first <= elapsed)))


def gate(rows, *, expected_pairs=9, mechanism_passed=False):
    """Missing/unstarted arms stay missing, never counted as observed failures."""
    pairs = {}
    for row in rows:
        if row.get("retry_of"):
            continue  # Original complete ledger remains primary; retries are sensitivity only.
        key = (row["task"], int(row["episode_index"]))
        if row["arm"] in pairs.setdefault(key, {}):
            raise ValueError("duplicate primary arm")
        pairs[key][row["arm"]] = row
    completed = [p for p in pairs.values() if set(p) == {"A", "B"}
                 and all(measured_terminal(r) for r in p.values())]
    result = {"status": "inconclusive", "expected_pairs": expected_pairs,
              "completed_pairs": len(completed), "mechanism_passed": mechanism_passed}
    counts = {arm: {"raw900": 0, "S300": 0} for arm in ("A", "B")}
    ratios = []
    for pair in completed:
        for arm, row in pair.items():
            raw = row.get("raw_libero_success") is True
            first = row.get("time_to_raw_success_s")
            counts[arm]["raw900"] += int(raw and finite_seconds(first) and first <= 900)
            counts[arm]["S300"] += int(score(raw, row.get("service_elapsed_s")))
        a, b = pair["A"], pair["B"]
        if (a.get("raw_libero_success") is True and b.get("raw_libero_success") is True
                and a["time_to_raw_success_s"] <= 900 and b["time_to_raw_success_s"] <= 900):
            av, bv = a.get("service_elapsed_s"), b.get("service_elapsed_s")
            if finite_seconds(av) and finite_seconds(bv) and av > 0:
                ratios.append((av - bv) / av)
    result.update(counts=counts, common_success_pairs=len(ratios),
                  median_paired_service_reduction=statistics.median(ratios) if ratios else None)
    if len(completed) != expected_pairs:
        result["reason"] = "required paired observations incomplete"
        return result
    speed = counts["B"]["S300"] >= counts["A"]["S300"] + 2
    speed |= (counts["B"]["S300"] >= counts["A"]["S300"] and len(ratios) >= 5
              and statistics.median(ratios) >= .20)
    quality = not any(r.get("new_reproducible_severe_quality_defect") for p in completed for r in p.values())
    passed = mechanism_passed and quality and speed and counts["B"]["raw900"] >= counts["A"]["raw900"]
    result.update(status="continue" if passed else "freeze", reason="preregistered engineering gate")
    return result


def summarize(root):
    root = Path(root)
    manifest = json.loads((root / "execution_manifest.json").read_text())
    stop_path = root / "online_stop.json"
    stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
    unstarted_p5_reason = (stop.get("reason") or stop.get("decision") or
                           "not yet reserved; inspect recorded stage gates and individual review requirements")
    reservations = [json.loads(p.read_text()) for p in sorted((root / "reservations").glob("trial_*.json"))
                    if p.stem.removeprefix("trial_").isdigit()]
    artifacts = {}
    for path in (root / "evaluations").glob("*/evidence_result.json"):
        row = json.loads(path.read_text())
        details = {}
        annotation = path.parent / "infrastructure_annotation.json"
        if annotation.exists():
            details = json.loads(annotation.read_text())
            if details.get("reservation_id") != row["reservation_id"]:
                raise ValueError("infrastructure annotation reservation mismatch")
            row.update(infrastructure_failure=details.get("infrastructure_failure") is True,
                       infrastructure_evidence=details.get("infrastructure_evidence"))
        row["reason"] = row.get("reason") or details.get("reason") or row.get("status")
        artifacts[row["reservation_id"]] = dict(row, evidence_path=str(path.relative_to(root)))
    sensitivity = {}
    for stage, filename in (("P5", "paired_results.csv"), ("P6", "heldout_results.csv")):
        rows = []
        for condition in manifest["splits"][stage]:
            for arm in ("A", "B"):
                match = [r for r in reservations if r["stage"] == stage and r["task"] == condition["task"]
                         and r["episode_index"] == condition["episode_index"] and r["arm"] == arm]
                if not match:
                    rows.append(dict(condition, arm=arm, status="not_started", reason=(
                        "P5 not passed; holdout remains sealed" if stage == "P6" else
                        unstarted_p5_reason)))
                for item in match:
                    row = dict(condition, arm=arm, reservation_id=item["reservation_id"],
                               retry_of=item.get("retry_of"), status="awaiting_review",
                               reason="reserved; inspect session evidence for preparation status")
                    row.update(artifacts.get(item["reservation_id"], {}))
                    if measured_terminal(row):
                        row["S300"] = score(row["raw_libero_success"], row["service_elapsed_s"])
                    rows.append(row)
        fields = ["task", "episode_index", "seed", "arm", "reservation_id", "retry_of", "status",
                  "raw_libero_success", "time_to_raw_success_s", "service_elapsed_s", "S300",
                  "autonomous_quality", "approved_by", "review_modified", "human_review_wait_s",
                  "automatic_review_compute_s", "reason", "evidence_path"]
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        (root / (stage.lower() + "_gate.json")).write_text(json.dumps(gate(
            rows, mechanism_passed=manifest.get("stage_gates", {}).get("P2") == "passed"), indent=2))
        excluded = {(r["task"], r["episode_index"]) for r in rows
                    if r.get("infrastructure_failure") is True and r.get("infrastructure_evidence")}
        retained = [r for r in rows if (r["task"], r["episode_index"]) not in excluded]
        stats = gate(retained, mechanism_passed=False)
        sensitivity[stage] = {"excluded_conditions": [list(p) for p in sorted(excluded)],
                             "retained_complete_pairs": stats["completed_pairs"],
                             "counts": stats["counts"],
                             "common_success_pairs": stats["common_success_pairs"],
                             "median_paired_service_reduction": stats["median_paired_service_reduction"],
                             "not_an_advancement_gate": True,
                             "rule": "Exclude an entire pair only for explicitly evidenced infrastructure failure; preserve original ledger and primary results."}
    (root / "sensitivity_analysis.json").write_text(json.dumps(sensitivity, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    take = sub.add_parser("reserve")
    take.add_argument("--stage", choices=["P4", "P5", "P6"], required=True)
    take.add_argument("--task", required=True)
    take.add_argument("--episode-index", type=int, required=True)
    take.add_argument("--arm", choices=["A", "B"], required=True)
    take.add_argument("--retry-of")
    take.add_argument("--reason")
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "reserve":
        print(reserve(args.root, stage=args.stage, task=args.task, episode_index=args.episode_index,
                      arm=args.arm, retry_of=args.retry_of, reason=args.reason))
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
