"""Run one fresh bowl episode with no action limit and a finite API budget.

Stop on LIBERO success, user stop, API budget reached, or infrastructure error.
There is no policy-action, total API-call, or elapsed-time cutoff. Each HTTP
request has its own timeout and output-token limit. Ctrl-C stops the run.
Resume validates provenance, preserves recorded terminal outcomes, and restarts
an unrecorded interrupted episode from its official initial state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from gpt_field.astra import APIError, AstraClient, AstraConfig, urllib_transport
from gpt_field.experiment import run_libero, write_json
from gpt_field.libero import LiberoConfig
from gpt_field.planner import PlannerConfig
from gpt_field.policy import FieldPolicy
from gpt_field.schema import GridSpec


ROOT = Path(__file__).resolve().parents[1]
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
STOPPING_RULE = ["LIBERO success", "user stop", "API budget reached", "infrastructure error"]
DEFAULT_SIM_PYTHON = "/workspace/venvs/libero-plus/bin/python"


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, (list, tuple)):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    return value


def redacting_transport(secret, base_transport=None):
    """Sanitize response bodies and transport exceptions before API logging."""
    base_transport = base_transport or urllib_transport

    def send(endpoint, payload, headers, timeout_s):
        try:
            return redact(base_transport(endpoint, payload, headers, timeout_s), secret)
        except APIError as exc:
            raise APIError(redact(str(exc), secret),
                           response_body=redact(exc.response_body, secret)) from None
        except Exception as exc:
            raise APIError(redact(f"Transport {type(exc).__name__}: {exc}", secret)) from None

    return send


class BowlFieldPolicy(FieldPolicy):
    def act(self, observation):
        if observation.prompt != TASK:
            raise ValueError("Task text does not match the requested bowl task")
        return super().act(observation)


def validate_config(config):
    if config.model != "gpt-6-astra":
        raise ValueError("This episode requires gpt-6-astra")
    for name in ("max_api_calls", "max_wall_time_s"):
        if getattr(config, name) is not None:
            raise ValueError(f"Unbounded episode requires {name}=null")
    cost = config.max_cost_usd
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("Episode requires a finite positive max_cost_usd")
    if config.max_output_tokens != 8192 or config.timeout_s != 180:
        raise ValueError("Per-request limits must be 8192 output tokens and 180 seconds")
    rates = (config.input_price_per_million, config.cached_input_price_per_million,
             config.cache_write_price_per_million, config.output_price_per_million)
    if rates != (10, 1, 12.5, 50):
        raise ValueError("Usage accounting rates must be input=10, cached=1, cache-write=12.5, output=50")


def reference_metadata(config):
    return {
        "runner": {"path": "scripts/evaluate_bowl_until_success.py",
                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "task": TASK,
        "stopping_rule": STOPPING_RULE,
        "policy": {"class": "evaluate_bowl_until_success.BowlFieldPolicy",
                   "exact_task_prompt_guard": True, "method": "field", "seed": 7,
                   "grid": GridSpec(size=9, spacing_m=0.02).to_dict(),
                   "planner": asdict(PlannerConfig(step_m=0.01, horizon=3)),
                   "update_steps": 5},
        "episode_limits": {"max_steps": None, "max_api_calls": None,
                           "max_wall_time_s": None, "max_cost_usd": config.max_cost_usd},
        "resume_policy": "Preserve terminal rows; unrecorded interrupted episodes restart from reset with their API ledger.",
    }


def read_progress(output):
    """Read evaluator artifacts without touching simulator or policy internals."""
    output = Path(output)
    try:
        state = json.loads((output / "run_state.json").read_text())
    except (OSError, ValueError):
        state = {}
    budget = state.get("budget") or {}
    return {"status": state.get("status", "starting"),
            "step": state.get("step"), "stage": state.get("stage"),
            "api_calls_recorded": budget.get("api_calls", 0),
            "raw_request_count": sum(1 for _ in (output / "api").glob("*.request.json")),
            "cost_usd_upper_bound": budget.get("cost_usd_upper_bound")}


class ProgressMonitor:
    """Progress remains responsive while the foreground thread awaits HTTP."""

    def __init__(self, output, *, interval_s=30):
        self.output = Path(output)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._watch, name="episode-progress", daemon=True)

    def emit(self, event):
        print(json.dumps({"event": event, **read_progress(self.output),
                          "elapsed_s": round(time.monotonic() - self.started, 1)}), flush=True)

    def _watch(self):
        while not self.stop_event.wait(self.interval_s):
            self.emit("progress")

    def start(self):
        self.emit("started")
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)


def episode_summary(output, episode_index, *, error=None, user_stopped=False):
    """Infrastructure errors and user stops are incomplete, never scored failures."""
    output = Path(output)
    row = None
    rows_path = output / "episodes.jsonl"
    try:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        if len(rows) != 1 or rows[0].get("task_id") != 0 or rows[0].get("episode_index") != episode_index:
            raise ValueError("Expected one result for the requested task and initial state")
        row = rows[0]
    except FileNotFoundError:
        pass
    except (ValueError, TypeError, KeyError) as exc:
        error = error or f"Result record error: {exc}"
    success = error is None and row is not None and row.get("status") == "success"
    stopped = user_stopped or (row is not None and row.get("status") in ("user_stop", "interrupted"))
    budget_exceeded = row is not None and row.get("status") == "budget_exceeded"
    status = ("success" if success else "user_stop" if stopped else
              "budget_exceeded" if budget_exceeded else "infrastructure_error")
    return {"status": status, "task": TASK, "suite": "libero_spatial", "task_id": 0,
            "episode_index": episode_index, "requested_episodes": 1,
            "stopping_rule": STOPPING_RULE, "evaluation_completed": success,
            "libero_success": True if success else None,
            "steps": row.get("steps") if row else read_progress(output)["step"],
            "episode_status": row.get("status") if row else None,
            "video": row.get("video") if row else None,
            "error": error or (row.get("error") if row else None) or (
                "No successful terminal episode record" if status == "infrastructure_error" else None),
            "scope": "One episode with no action, total API-call, or elapsed-time cutoff, subject to a finite API cost budget. Incomplete runs are not scored as task failures."}


def execute_episode(output, *, config, sim_config, api_key, episode_index=0, resume=False):
    validate_config(config)
    if sim_config.suite != "libero_spatial" or sim_config.seed != 7 or sim_config.max_steps is not None:
        raise ValueError("Simulator requires libero_spatial, seed 7, max_steps=None")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    # Remove an inherited key before the evaluator can spawn a simulator process.
    os.environ.pop("OPENAI_API_KEY", None)
    output = Path(output)
    if resume:
        if not (output / "manifest.json").is_file():
            raise ValueError("resume requires an existing run manifest")
    elif output.exists():
        raise FileExistsError("Fresh episode requires a new output directory; use --resume for an existing run")
    client = AstraClient(config, api_key=api_key, transport=redacting_transport(api_key), log_dir=output / "api")
    client.preflight()
    policy = BowlFieldPolicy(client, method="field", grid=GridSpec(size=9, spacing_m=0.02),
                             planner_config=PlannerConfig(step_m=0.01, horizon=3), update_steps=5, seed=7)
    monitor = ProgressMonitor(output)
    failure, user_stopped = None, False
    monitor.start()
    try:
        run_libero(output, policy, sim_config, task_ids=[0], episode_indices=[episode_index],
                   client=client, phase="exploratory", resume=resume, save_video=True,
                   reference_metadata=reference_metadata(config))
    except KeyboardInterrupt:
        user_stopped = True
    except Exception as exc:
        failure = redact(f"{type(exc).__name__}: {exc}", api_key)
    finally:
        monitor.close()
    summary = redact(episode_summary(output, episode_index, error=failure, user_stopped=user_stopped), api_key)
    summary["budget"] = client.budget_snapshot()
    summary["max_cost_usd"] = config.max_cost_usd
    summary["output"] = str(output)
    if output.is_dir():
        write_json(output / "runner_summary.json", summary)
    print(json.dumps({"event": "finished", **summary}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", "/workspace/heading_flow_pub/third_party/LIBERO"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--api-config", default=str(ROOT / "configs/astra_unbounded_episode.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = AstraConfig(**json.loads(Path(args.api_config).read_text()))
    validate_config(config)
    # abspath preserves the virtualenv symlink; resolve() would select base Python.
    sim_config = LiberoConfig(suite="libero_spatial", seed=7, max_steps=None,
                              libero_root=str(Path(args.libero_root).resolve()),
                              python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
    key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    result = execute_episode(args.output, config=config, sim_config=sim_config, api_key=key,
                             episode_index=args.episode_index, resume=args.resume)
    return (0 if result["status"] == "success" else 130 if result["status"] == "user_stop"
            else 3 if result["status"] == "budget_exceeded" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
