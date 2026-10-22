"""Frozen, concurrent evaluation of one LIBERO task on official init indices 0–9.

Run from the project environment. Credentials remain in this process only.
Resume keeps recorded outcomes immutable and continues only unfinished children.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import threading
import time

from gpt_field.astra import APIError, AstraClient, AstraConfig, urllib_transport
from gpt_field.experiment import digest, run_libero, source_manifest, write_json
from gpt_field.libero import LiberoConfig
from gpt_field.planner import PlannerConfig
from gpt_field.policy import FieldPolicy
from gpt_field.report import wilson
from gpt_field.schema import GridSpec


TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
INDICES = tuple(range(10))
MAX_STEPS = 220
ROOT = Path(__file__).resolve().parents[1]


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    return value


def fatal_account_error(message, body=None):
    text = (str(message) + " " + str(body or "")).lower()
    return any(marker in text for marker in (
        "http 401", "insufficient_quota", "invalid_api_key", "authentication_error",
        "billing_hard_limit", "billing_not_active", "quota_exceeded",
    ))


class DispatchGate:
    """Stop new HTTP dispatches after an authentication/quota failure."""

    def __init__(self):
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.reason = None
        self.dispatched = 0

    def stop(self, reason):
        with self.lock:
            if not self.stopped.is_set():
                self.reason = reason
                self.stopped.set()

    def transport(self, secret, base_transport=None):
        base_transport = base_transport or urllib_transport

        def send(endpoint, payload, headers, timeout_s):
            with self.lock:
                if self.stopped.is_set():
                    raise APIError("Batch dispatch stopped: " + str(self.reason))
                self.dispatched += 1
            try:
                response = redact(base_transport(endpoint, payload, headers, timeout_s), secret)
            except APIError as exc:
                message, body = redact(str(exc), secret), redact(exc.response_body, secret)
                if fatal_account_error(message, body):
                    self.stop("authentication or quota failure; inspect preserved API error logs")
                raise APIError(message, response_body=body) from None
            except Exception as exc:
                # Do not allow an unexpected transport exception to persist a credential.
                raise APIError(redact(f"Transport {type(exc).__name__}: {exc}", secret)) from None
            if isinstance(response, dict) and response.get("error"):
                if fatal_account_error("", json.dumps(response["error"])):
                    self.stop("authentication or quota failure; inspect preserved API error logs")
            return response

        return send


class BowlFieldPolicy(FieldPolicy):
    def act(self, observation):
        if observation.prompt != TASK:
            raise ValueError("Task text does not match the frozen bowl evaluation")
        return super().act(observation)


def validate_config(config):
    if config.model != "gpt-6-astra":
        raise ValueError("This evaluation requires gpt-6-astra")
    if config.max_api_calls is None or not 1 <= config.max_api_calls <= 88:
        raise ValueError("Per-episode max_api_calls must be between 1 and 88")
    if config.max_output_tokens > 8192 or config.timeout_s > 180:
        raise ValueError("Output/request limits cannot exceed 8192 tokens / 180 seconds")
    if config.max_wall_time_s is None or not 0 < config.max_wall_time_s <= 18000:
        raise ValueError("Per-episode wall-time cap must be in (0, 18000]")
    if config.max_cost_usd is None or not 0 < config.max_cost_usd <= 50:
        raise ValueError("Per-episode accounting cap must be in (0, 50]")
    expected = (10, 1, 12.5, 50)
    actual = (config.input_price_per_million, config.cached_input_price_per_million,
              config.cache_write_price_per_million, config.output_price_per_million)
    if actual != expected:
        raise ValueError("Frozen accounting rates must be input=10, cached=1, cache-write=12.5, output=50")


def _read(path):
    return json.loads(path.read_text())


def episode_directory(output, index):
    return Path(output) / f"episode_{index:02d}"


def read_episode(output, index):
    """One intended initial state is always exactly one aggregate row."""
    directory = episode_directory(output, index)
    result = {"episode_index": index, "task_id": 0, "status": "pending",
              "directory": directory.name, "evaluation_completed": False}
    rows_path = directory / "episodes.jsonl"
    if rows_path.exists():
        try:
            rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            if len(rows) > 1:
                raise ValueError("more than one result for the requested initial state")
            if rows:
                row = rows[0]
                if row.get("episode_index") != index or row.get("task_id") != 0:
                    raise ValueError("child task/initial-state identity mismatch")
                result.update(row)
                result["evaluation_completed"] = row["status"] == "success" or (
                    row["status"] == "timeout" and row.get("steps") == MAX_STEPS)
                if row["status"] == "timeout" and not result["evaluation_completed"]:
                    result["status"] = "incomplete_timeout"
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status="record_error", error=str(exc))
    state_path = directory / "run_state.json"
    if state_path.exists():
        try:
            state = _read(state_path)
            result["budget"] = state.get("budget")
            if state.get("status") == "running":
                result["current_step"] = state.get("step")
                result["current_stage"] = state.get("stage")
            if result["status"] == "pending":
                if state.get("status") in ("failed", "interrupted"):
                    result.update(status=state["status"], error=state.get("error"))
                elif state.get("status") == "running":
                    result["status"] = "running"
        except (ValueError, TypeError) as exc:
            result.update(status="record_error", error=str(exc))
    coordinator_path = directory / "coordinator_result.json"
    if coordinator_path.exists() and result["status"] in ("pending", "running", "failed", "interrupted"):
        result.update(_read(coordinator_path))
    return result


def aggregate(output, *, gate=None, active=(), elapsed_s=0):
    episodes = [read_episode(output, index) for index in INDICES]
    for row in episodes:
        if row["episode_index"] in active and not row["evaluation_completed"] and row["status"] == "pending":
            row["status"] = "running"
    completed = [row for row in episodes if row["evaluation_completed"]]
    success_count = sum(row["status"] == "success" for row in completed)
    infra = [row for row in episodes if row["status"] not in ("pending", "running")
             and not row["evaluation_completed"]]
    pending = [row for row in episodes if row["status"] in ("pending", "running")]
    final = len(completed) == len(INDICES)
    budgets = [row["budget"] for row in episodes if isinstance(row.get("budget"), dict)]
    costs = [budget.get("cost_usd_upper_bound") for budget in budgets]
    summary = {
        "status": "complete" if final else "provisional",
        "task": TASK, "suite": "libero_spatial", "task_id": 0,
        "requested_episode_indices": list(INDICES), "intended_denominator": len(INDICES),
        "completed_evaluations": len(completed), "successes": success_count,
        "task_failures": sum(row["status"] == "timeout" for row in completed),
        "infrastructure_failures": len(infra), "pending_or_running": len(pending),
        # Never turn infrastructure failure or missing evaluations into a zero score.
        "success_rate": success_count / len(INDICES) if final else None,
        "wilson_95": wilson(success_count, len(INDICES)) if final else None,
        "observed_completed_success_rate": success_count / len(completed) if completed else None,
        "api_calls_reserved": sum(budget.get("api_calls", 0) for budget in budgets),
        "cost_usd_upper_bound": sum(costs) if all(isinstance(v, (int, float)) for v in costs) else None,
        "elapsed_s": elapsed_s, "episodes": episodes,
        "dispatch_stopped": gate.stopped.is_set() if gate else False,
        "dispatch_stop_reason": gate.reason if gate else None,
        "http_dispatches_this_process": gate.dispatched if gate else None,
        "scope": "One task, ten fixed official initial states, paused simulation. A final score requires ten success/220-step timeout outcomes. Infrastructure errors are not task failures.",
    }
    return summary


def write_gallery(output, summary):
    cards = []
    for row in summary["episodes"]:
        media = "<p>No recorded video yet.</p>"
        # Child runner may expose video or GIF artifacts; only link local paths.
        video = row.get("video")
        if video:
            path = Path(row["directory"]) / video
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Video path must remain inside the local run directory")
            url = html.escape(path.as_posix(), quote=True)
            if path.suffix.lower() == ".gif":
                media = f'<img src="{url}" loading="lazy" alt="Episode {row["episode_index"]} rollout">'
            else:
                media = f'<video controls preload="metadata" src="{url}"></video>'
        cards.append(f'<article><h2>Initial state {row["episode_index"]}: {html.escape(row["status"])}</h2>'
                     f'<p>{row.get("steps", 0)} policy steps</p>{media}</article>')
    rate = "Pending: provisional results" if summary["success_rate"] is None else f'{summary["successes"]}/10 successes'
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page += '<title>LIBERO bowl evaluation</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#17202a}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}article{padding:16px;background:#f2f4f7;border-radius:10px}h2{font-size:18px}img,video{width:100%;height:auto}</style>'
    page += f'<h1>LIBERO bowl evaluation</h1><p>{html.escape(TASK)}</p><p><strong>{rate}</strong></p>'
    page += f'<p>{summary["completed_evaluations"]}/10 completed evaluations; {summary["infrastructure_failures"]} infrastructure failures; {summary["pending_or_running"]} pending/running.</p>'
    page += '<p>Official initial states 0–9, task 0, 220 policy steps maximum. Simulation pauses for model requests.</p><p><a href="summary.json">Full results</a> · <a href="manifest.json">Frozen protocol</a></p><main>'
    page += "".join(cards) + "</main></html>"
    temporary = Path(output) / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(Path(output) / "index.html")


def _run_episode(output, index, config, sim_config, key, gate, resume):
    directory = episode_directory(output, index)
    if (directory / "episodes.jsonl").exists():
        # Let run_libero recover an interrupted final append when there is no terminal row.
        current = read_episode(output, index)
        if current["status"] not in ("pending", "running", "record_error", "interrupted"):
            return current
    elif (directory / "coordinator_result.json").exists():
        return read_episode(output, index)
    if gate.stopped.is_set():
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "coordinator_result.json", {
            "status": "aborted", "error": "Batch dispatch stopped before this episode started",
            "evaluation_completed": False,
        })
        return read_episode(output, index)
    client = AstraClient(config, api_key=key, transport=gate.transport(key), log_dir=directory / "api")
    policy = BowlFieldPolicy(client, method="field", grid=GridSpec(size=9, spacing_m=0.02),
                             planner_config=PlannerConfig(step_m=0.01, horizon=3),
                             update_steps=5, seed=7)
    try:
        run_libero(directory, policy, sim_config, task_ids=[0], episode_indices=[index],
                   client=client, phase="frozen", resume=resume and (directory / "manifest.json").exists(),
                   save_video=True)
    except Exception as exc:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "coordinator_result.json", {
            "status": "infrastructure_error", "error": redact(f"{type(exc).__name__}: {exc}", key),
            "evaluation_completed": False,
        })
    return read_episode(output, index)


def execute_batch(output, *, config, sim_config, api_key, workers=10, resume=False):
    if type(workers) is not int or not 1 <= workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    validate_config(config)
    if sim_config.suite != "libero_spatial" or sim_config.max_steps != MAX_STEPS or sim_config.seed != 7:
        raise ValueError("Frozen simulator requires libero_spatial, seed 7, 220 steps")
    # A supplied environment key must never reach simulator subprocesses.
    os.environ.pop("OPENAI_API_KEY", None)
    output = Path(output)
    identity = {
        "task": TASK, "suite": "libero_spatial", "task_id": 0, "episode_indices": list(INDICES),
        "phase": "frozen", "method": "field", "grid": GridSpec().to_dict(),
        "planner": asdict(PlannerConfig()), "update_steps": 5,
        "simulator": asdict(sim_config), "api_per_episode": asdict(config),
        "runtime_environment": {name: os.environ.get(name) for name in (
            "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "software": source_manifest(),
        "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    identity_hash = digest(identity)
    if resume:
        if _read(output / "manifest.json")["identity_hash"] != identity_hash:
            raise ValueError("resume rejected: frozen source/configuration changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "manifest.json", {
            "identity_hash": identity_hash, "configuration": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(), "workers": workers,
            "max_total_api_calls": 10 * config.max_api_calls,
            "max_total_accounting_usd": 10 * config.max_cost_usd,
            "simulation_pauses_for_api": True,
            "resume_policy": "Preserve all recorded outcomes including infrastructure failures; only interrupted/unrecorded children continue, with their existing API ledger.",
        })
    # Validate key and accounting before simulator construction, without sending HTTP.
    AstraClient(config, api_key=api_key).preflight()
    gate = DispatchGate()
    started = time.monotonic()

    def persist(active=()):
        summary = aggregate(output, gate=gate, active=active, elapsed_s=time.monotonic() - started)
        write_json(output / "summary.json", summary)
        write_gallery(output, summary)
        return summary

    def progress(event, summary, **extra):
        print(json.dumps({"event": event, "completed_evaluations": summary["completed_evaluations"],
                          "infrastructure_failures": summary["infrastructure_failures"],
                          "pending_or_running": summary["pending_or_running"],
                          "http_dispatches_this_process": gate.dispatched,
                          "elapsed_s": round(time.monotonic() - started, 1), **extra}), flush=True)

    persist()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bowl")
    futures = {}
    try:
        for index in INDICES:
            future = executor.submit(_run_episode, output, index, config, sim_config, api_key, gate, resume)
            futures[future] = index
        progress("started", persist(futures.values()), workers=workers)
        while futures:
            completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            if not completed:
                progress("progress", persist(futures.values()))
            for future in completed:
                index = futures.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    directory = episode_directory(output, index)
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory / "coordinator_result.json", {
                        "status": "infrastructure_error", "error": redact(f"{type(exc).__name__}: {exc}", api_key),
                        "evaluation_completed": False,
                    })
                    row = read_episode(output, index)
                progress("episode_finished", persist(futures.values()), episode_index=index, status=row["status"])
    except BaseException:
        gate.stop("coordinator interrupted")
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        summary = persist()
        progress("finished", summary, status=summary["status"], output=str(output))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sim-python", default=sys.executable)
    parser.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", "/workspace/heading_flow_pub/third_party/LIBERO"))
    parser.add_argument("--api-config", default=str(ROOT / "configs/astra_bowl10.json"))
    args = parser.parse_args(argv)
    config = AstraConfig(**_read(Path(args.api_config)))
    validate_config(config)
    key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    sim_config = LiberoConfig(suite="libero_spatial", seed=7, max_steps=MAX_STEPS,
                               libero_root=str(Path(args.libero_root).resolve()),
                               python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
    result = execute_batch(args.output, config=config, sim_config=sim_config, api_key=key,
                           workers=args.workers, resume=args.resume)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
