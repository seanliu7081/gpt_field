"""Robot-only finger-gap traces for the provisional G7 / G8 floors and sigma_g (spec §5.2, M0; zero model calls).

Open and closed-EMPTY hands move through free space (no object contact): lateral carries (+-x, +-y at 1 cm/step),
lowerings (-z at 5 mm/step) and returns, yaws about the approach (0.1 rad/step, both senses) and a static hold.
Every step's finger gap (q1 - q2) is recorded. The generic contact monitor runs; a segment with any robot contact
other than robot-internal is discarded. Floors follow the spec rule (robot-only inputs only; task outcomes are
never an input):

  floor = max(2 x the robot-only p99, 1.5 x the recorded v1 value)

- G7 total held-contact rise during carries: recorded 0.55 mm (v1 clean carries); per step: recorded 0.149 mm;
- G8 per-step rise while lowering: recorded 0.09 mm;
- sigma_g: the closed-empty-hand gap-rate noise under yaw (K1w / K4w input), floored at max(the open hand's
  static per-step p99, the 1e-4 mm measurement resolution): a closed EMPTY hand rests on its hard stop, so its
  own noise is 0 and cannot be a divisor;
- G9 open-finger drift on free reaches: v1 states <= 0.55 mm.

Every floor is reported as a DD record (label, value and derivation, citing the recorded v1 value's source
file:line). When the robot-only term 2 x p99 is below 1.5 x the recorded value it is non-binding and the floor
is the 1.5 x recorded value.

One fresh worker process per task key (episode 0 by default). Diagnosis only: the floors are REPORTED; a DD
change is a version bump made in the module that owns the threshold.

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python -B scripts/agent_speedup_v2_robot_traces.py \
           --out runs/agent_speedup_v2_sim_smoke_m0_robot_traces [--keys libero_10_t2,bowl_plate,drawer_bowl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RECORDED = {"G7_total_mm": 0.55, "G7_per_step_mm": 0.149, "G8_per_step_mm": 0.09, "G9_open_drift_mm": 0.55}
RECORDED_SOURCES = {
    "G7_total_mm": "src/gpt_field/contact_skills.py:886 (GAP_RISE_MM comment: live 001 clean carries <= 0.55 mm)",
    "G7_per_step_mm": "src/gpt_field/contact_skills.py:887 (GAP_STEP_RISE_MM comment: free-carry noise <= 0.149 "
                      "mm/step)",
    "G8_per_step_mm": "src/gpt_field/agent_speedup_v1.py:62 (PLACE_GAP_RISE_MM comment: free lowering drift <= "
                      "0.09 mm)",
    "G9_open_drift_mm": "src/gpt_field/contact_skills.py:891 (FINGER_SETTLED_MM comment: free reaches drift <= "
                        "0.55 mm)",
}
# Settle before measuring. The open gap keeps ramping long after the reset's open settle (77.7 -> 79.9 mm over 80
# steps, still +0.003 mm/step): a fixed 80-step prep, the remaining ramp is part of the open drift pool. An empty
# close is still falling ~0.05 mm/step at 30 steps and reaches its hard stop a few steps later, so it settles until
# |d gap| < CLOSED_SETTLE_RATE_MM per step (at least 30, at most 80 steps). Both ramps are reported (prep_gap_mm).
PREP_STEPS = {"open": 80, "closed_empty": 30}
PREP_MAX_STEPS = 80
CLOSED_SETTLE_RATE_MM = 1e-3
MEASUREMENT_RESOLUTION_MM = 1e-4
PROVISIONAL = {"G7_total_mm": 1.0, "G7_per_step_mm": 0.4, "G8_per_step_mm": 0.3, "G9_open_drift_mm": 1.0}


def segments(grip):
    """(name, kind, actions); grip -1 open / +1 closed-empty."""
    a = lambda dx=0.0, dy=0.0, dz=0.0, yaw=0.0: [dx, dy, dz, 0.0, 0.0, yaw, grip]
    return [
        ("static", "static", [a()] * 10),
        ("carry_+x", "carry", [a(dx=0.2)] * 20), ("carry_-x", "carry", [a(dx=-0.2)] * 20),
        ("carry_+y", "carry", [a(dy=0.2)] * 20), ("carry_-y", "carry", [a(dy=-0.2)] * 20),
        ("lower", "lower", [a(dz=-0.1)] * 10), ("raise", "raise", [a(dz=0.1)] * 10),
        ("yaw_+", "yaw", [a(yaw=0.2)] * 12), ("yaw_-", "yaw", [a(yaw=-0.2)] * 24), ("yaw_+back", "yaw", [a(yaw=0.2)] * 12),
        ("static_end", "static", [a()] * 10),
    ]


def single(key, episode):
    from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
    from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config, resolve_v2_task
    from dataclasses import replace
    spec = resolve_v2_task(key)
    config, monitor = build_v2_sim_config(key)
    config = replace(config, stop_on_success=False, render_size=128, image_size=128)
    worker = V2LiberoWorker(config, {**monitor, "truth_window": False})
    out = {"key": key, "episode": episode, "hands": {}}
    try:
        for hand, grip in (("open", -1.0), ("closed_empty", 1.0)):
            response = worker.reset(spec.task_id, episode)
            prep_gaps = []
            while True:
                response = worker.step([0, 0, 0, 0, 0, 0, grip])
                s = response["observation"]["observation/state"]
                prep_gaps.append(round((s[6] - s[7]) * 1000.0, 5))
                n = len(prep_gaps)
                if hand == "open" and n >= PREP_STEPS["open"]:
                    break
                if hand == "closed_empty" and n >= PREP_STEPS["closed_empty"] and (
                        abs(prep_gaps[-1] - prep_gaps[-2]) < CLOSED_SETTLE_RATE_MM or n >= PREP_MAX_STEPS):
                    break
            out.setdefault("prep_gap_mm", {})[hand] = prep_gaps
            records = []
            for name, kind, actions in segments(grip):
                before = dict(worker.generic.counts)
                gaps, poses = [], []
                for action in actions:
                    response = worker.step(action)
                    s = response["observation"]["observation/state"]
                    gaps.append((s[6] - s[7]) * 1000.0)
                    poses.append(list(s[:6]))
                touched = {k: worker.generic.counts[k] - before[k] for k in worker.generic.counts
                           if k.startswith("points_") and k not in ("points_robot_internal", "points_passive_scene")
                           and worker.generic.counts[k] != before[k]}
                records.append({"segment": name, "kind": kind, "gap_mm": [round(g, 5) for g in gaps],
                                "first_pose": poses[0], "last_pose": poses[-1], "contacts": touched})
            out["hands"][hand] = records
    finally:
        worker.close()
    return out


def summarize(rows):
    import numpy as np

    def p99(values):
        return float(np.percentile(np.abs(values), 99)) if len(values) else None
    pools = {k: [] for k in ("carry_total", "carry_step", "lower_step", "yaw_rate", "open_drift", "static_step",
                             "open_static_step")}
    discarded = []
    for row in rows:
        for hand, records in row["hands"].items():
            for rec in records:
                if rec["contacts"]:
                    discarded.append([row["key"], hand, rec["segment"], rec["contacts"]])
                    continue
                g = np.asarray(rec["gap_mm"])
                d = np.diff(g)
                if hand == "closed_empty":
                    if rec["kind"] == "carry":
                        pools["carry_total"].extend((g - g[0]).tolist())
                        pools["carry_step"].extend(d.tolist())
                    elif rec["kind"] == "lower":
                        pools["lower_step"].extend(d.tolist())
                    elif rec["kind"] == "yaw":
                        pools["yaw_rate"].extend(d.tolist())
                    elif rec["kind"] == "static":
                        pools["static_step"].extend(d.tolist())
                elif rec["kind"] in ("carry", "lower", "raise", "yaw"):
                    pools["open_drift"].extend((g - g[0]).tolist())
                elif rec["kind"] == "static":
                    pools["open_static_step"].extend(d.tolist())
    stats = {k: {"n": len(v), "p99_abs_mm": None if not v else round(p99(v), 4),
                 "max_abs_mm": None if not v else round(float(np.max(np.abs(v))), 4),
                 "std_mm": None if not v else round(float(np.std(v)), 4)} for k, v in pools.items()}
    records = {}
    for name, pool in (("G7_total_mm", "carry_total"), ("G7_per_step_mm", "carry_step"),
                       ("G8_per_step_mm", "lower_step"), ("G9_open_drift_mm", "open_drift")):
        p = stats[pool]["p99_abs_mm"]
        if p is None:
            records[name] = {"value": None, "label": "DD", "derivation": f"no robot-only samples in {pool}"}
            continue
        robot_only, recorded = round(2 * p, 4), round(1.5 * RECORDED[name], 4)
        binding = robot_only > recorded
        records[name] = {
            "value": max(robot_only, recorded), "label": "DD", "pool": pool, "robot_only_2x_p99_mm": robot_only,
            "recorded_1p5x_mm": recorded, "robot_only_term_binding": binding,
            "derivation": (f"robot-only term binding (2 x p99 = {robot_only} mm > 1.5 x recorded v1 value "
                           f"{RECORDED[name]} mm = {recorded} mm; {RECORDED_SOURCES[name]})" if binding else
                           f"robot-only term non-binding (2 x p99 = {robot_only} mm); 1.5 x recorded v1 value "
                           f"{RECORDED[name]} mm = {recorded} mm ({RECORDED_SOURCES[name]})")}
    yaw_std, open_static = stats["yaw_rate"]["std_mm"], stats["open_static_step"]["p99_abs_mm"]
    sigma = max(v for v in (yaw_std, open_static, MEASUREMENT_RESOLUTION_MM) if v is not None)
    records["sigma_g_mm_per_step"] = {
        "value": sigma, "label": "DD", "closed_yaw_rate_std_mm": yaw_std, "open_static_step_p99_mm": open_static,
        "measurement_resolution_mm": MEASUREMENT_RESOLUTION_MM,
        "derivation": f"max(closed-empty yaw gap-rate std {yaw_std}, open static per-step p99 {open_static}, "
                      f"measurement resolution {MEASUREMENT_RESOLUTION_MM}) mm/step: a closed empty hand rests on "
                      "its hard stop, so its own noise is not a usable sigma"}
    floors = {name: record["value"] for name, record in records.items()}
    ramps = {hand: rows[0].get("prep_gap_mm", {}).get(hand) for hand in ("open", "closed_empty")}
    prep_steps = {f"{row['key']}:{hand}": len(row.get("prep_gap_mm", {}).get(hand) or [])
                  for row in rows for hand in ("open", "closed_empty")}
    closed_values = sorted({round(v, 3) for row in rows for rec in row["hands"]["closed_empty"]
                            for v in rec["gap_mm"]})
    return {"stats": stats, "derived_floors": floors, "floor_records": records, "settle_ramps_first_task_mm": ramps,
            "prep_steps": prep_steps, "prep_rule": {"open_fixed_steps": PREP_STEPS["open"],
                                                    "closed_empty_min_steps": PREP_STEPS["closed_empty"],
                                                    "closed_empty_max_steps": PREP_MAX_STEPS,
                                                    "closed_empty_settle_rate_mm": CLOSED_SETTLE_RATE_MM},
            "closed_empty_gap_values_mm": closed_values[:20],
            "note": "a closed EMPTY hand rests on the finger hard stop, so its gap carries no motion noise; "
                    "the closed-hand pools therefore bound robot-only noise only trivially and the floors fall "
                    "back to 1.5 x the recorded v1 values", "provisional_floors": PROVISIONAL,
            "recorded_v1_values": RECORDED, "recorded_v1_sources": RECORDED_SOURCES,
            "rule": "floor = max(2 x robot-only p99, 1.5 x recorded value) (spec §5.2 G7/G8)",
            "provisional_floors_cover_derived": {k: (floors[k] is None or PROVISIONAL[k] >= floors[k])
                                                 for k in PROVISIONAL},
            "discarded_segments_with_contact": discarded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--keys", default="libero_10_t2,bowl_plate,drawer_bowl")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--single")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args(argv)
    if args.single:
        print("ROW " + json.dumps(single(args.single, args.episode)), flush=True)
        return 0
    if args.out.exists():
        parser.error("out exists; choose a fresh directory")
    args.out.mkdir(parents=True)
    rows = []
    for key in args.keys.split(","):
        for episode in (int(e) for e in args.episodes.split(",")):
            proc = subprocess.run([sys.executable, "-B", __file__, "--single", key, "--episode", str(episode)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(proc.stderr[-3000:])
            rows.append(json.loads([l for l in proc.stdout.splitlines() if l.startswith("ROW ")][-1][4:]))
    (args.out / "traces.json").write_text(json.dumps(rows))
    summary = summarize(rows)
    (args.out / "robot_only_floors.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("derived_floors", "provisional_floors_cover_derived")} |
                     {"discarded": len(summary["discarded_segments_with_contact"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
