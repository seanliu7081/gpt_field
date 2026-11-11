"""Task-balanced reports with paired comparisons and a separate distance diagnostic."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z/total
    mid = (p + z*z/(2*total)) / d
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0.0, mid-radius), min(1.0, mid+radius)]


def summarize(records: list[dict], seed: int = 7) -> dict:
    if len({row["suite"] for row in records}) > 1:
        raise ValueError("report one benchmark suite at a time; task identifiers differ between suites")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["task_id"])].append(row)
    tasks = []
    for (method, task), rows in sorted(grouped.items(), key=lambda x: str(x[0])):
        keys = [(r["suite"], r["seed"], r["task_id"], r["episode_index"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate method/task/initial-state results; do not pool repeated runs silently")
        successes = sum(r["status"] == "success" for r in rows)
        tasks.append({"method": method, "task_id": task, "episodes": len(rows),
            "successes": successes, "success_rate": successes/len(rows),
            "wilson_95": wilson(successes, len(rows)), "statuses": dict(Counter(r["status"] for r in rows))})
    methods = {}
    rng = np.random.default_rng(seed)
    for method in sorted({r["method"] for r in records}):
        method_tasks = [t for t in tasks if t["method"] == method]
        samples = np.zeros(4000)
        for task in method_tasks:
            samples += rng.binomial(task["episodes"], task["success_rate"], size=4000) / task["episodes"]
        samples /= len(method_tasks)
        rows = [r for r in records if r["method"] == method]
        methods[method] = {
            "tasks_observed": len(method_tasks), "episodes": len(rows),
            "task_mean_success_rate": float(np.mean([t["success_rate"] for t in method_tasks])),
            "stratified_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "ci_note": "Within-task episode bootstrap; fixed observed task set; boundary rates can give degenerate intervals. See per-task Wilson intervals.",
            "mean_steps": float(np.mean([r.get("steps", 0) for r in rows])),
            "mean_path_length_m": float(np.mean([r.get("path_length_m", 0) for r in rows])),
            "statuses": dict(Counter(r["status"] for r in rows)),
        }
    paired = []
    names = sorted(methods)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            def keyed(method):
                return {(r["suite"], r["seed"], r["task_id"], r["episode_index"],
                         r.get("protocol_hash", "unspecified")): r["status"] == "success"
                        for r in records if r["method"] == method}
            left, right = keyed(a), keyed(b)
            shared = sorted(set(left) & set(right), key=str)
            if shared:
                by_task = defaultdict(list)
                for key in shared:
                    by_task[key[:3]].append(int(left[key])-int(right[key]))
                delta = float(np.mean([np.mean(v) for v in by_task.values()]))
                boot = np.zeros(4000)
                for differences in by_task.values():
                    boot += rng.choice(differences, size=(4000, len(differences)), replace=True).mean(axis=1)
                boot /= len(by_task)
                paired.append({"a": a, "b": b, "pairs": len(shared), "task_mean_difference_a_minus_b": delta,
                               "paired_bootstrap_95": np.quantile(boot, [0.025,0.975]).tolist(),
                               "unmatched_a": len(left)-len(shared), "unmatched_b": len(right)-len(shared)})
    return {"per_task": tasks, "methods": methods, "paired": paired,
            "scope": "Only observed tasks/episodes; incomplete suites are not full benchmark results."}


def load_results(paths: list[str | Path]) -> list[dict]:
    records = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "episodes.jsonl"
        with path.open() as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def distance_fidelity(data: dict) -> dict:
    """Only auxiliary datasets with explicitly supplied metric references are eligible."""
    if data.get("semantics") != "distance_m":
        raise ValueError("distance fidelity requires semantics='distance_m'; planning potentials are not calibrated distances")
    predicted = np.asarray(data["predicted"], dtype=float)
    reference = np.asarray(data["reference"], dtype=float)
    if predicted.ndim != 1 or reference.shape != predicted.shape or not len(predicted):
        raise ValueError("predicted and reference must be equally sized nonempty vectors")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all() or (predicted < 0).any() or (reference < 0).any():
        raise ValueError("distance arrays must be finite and nonnegative")
    errors = predicted-reference
    return {"samples": len(predicted), "mae_m": float(np.abs(errors).mean()),
            "rmse_m": float(np.sqrt((errors**2).mean())), "bias_m": float(errors.mean()),
            "scope": "Supplied auxiliary reference only; not a full manipulation task distance."}
