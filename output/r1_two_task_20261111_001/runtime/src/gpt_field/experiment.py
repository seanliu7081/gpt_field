"""Evaluator-owned run state, provenance, episode logs, and restart handling."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
import time
from itertools import count
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .astra import APIError, BudgetExceeded, MissingCredentials, SchemaError
from .libero import LiberoConfig, SimProcess, SimulatorError
from .report import summarize


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_manifest() -> dict:
    directory = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.py"))}
    packages = {}
    for name in ("numpy", "Pillow", "gpt-field"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages, "source_sha256": hashes}


def _read_episode_records(path: Path, recover_final: bool = False) -> list[dict]:
    raw = path.read_bytes()
    parts = raw.splitlines(keepends=True)
    rows = []
    for index, part in enumerate(parts):
        try:
            rows.append(json.loads(part))
        except (ValueError, UnicodeDecodeError):
            if not (recover_final and index==len(parts)-1 and not part.endswith(b"\n")):
                raise ValueError("episode results contain corruption before a recoverable final append")
            # Retain the exact damaged artifact; only the uncommitted last record is removed.
            backup = path.with_name(path.name+f".interrupted-{time.time_ns()}")
            backup.write_bytes(raw)
            path.write_bytes(b"".join(parts[:index]))
    # A valid final JSON object may have been persisted before its terminating newline.
    if recover_final and rows and path.read_bytes() and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def run_libero(output, policy, sim_config: LiberoConfig, *, task_ids: list[int],
               episode_indices: list[int], client=None, phase: str = "exploratory",
               resume: bool = False, save_video: bool = True, reference_metadata: dict | None = None):
    output = Path(output)
    if resume:
        if not (output / "manifest.json").is_file():
            raise ValueError("resume requires an existing run manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
    state_path = output / "run_state.json"
    method = getattr(policy, "method", "pi05")
    identity = {
        "simulator": asdict(sim_config), "method": method,
        "task_ids": task_ids, "episode_indices": episode_indices, "phase": phase,
        "grid": policy.grid.to_dict() if hasattr(policy,"grid") else None,
        "planner": asdict(policy.planner.config) if hasattr(policy,"planner") else None,
        "update_steps": policy.update_steps,
        "policy_seed": getattr(policy,"seed",None),
        "api": asdict(client.config) if client else None, "software": source_manifest(),
        "reference": reference_metadata,
        "video": "annotated_mp4_v1" if save_video else None,
    }
    identity_hash = digest(identity)
    if resume:
        old = json.loads((output / "manifest.json").read_text())
        if old["identity_hash"] != identity_hash:
            raise ValueError("resume rejected: source, configuration, or runtime changed")
        if client:
            if state_path.exists():
                snapshot = json.loads(state_path.read_text()).get("budget")
                if snapshot:
                    client.load_budget_snapshot(snapshot)
            client.reconcile_request_logs()
    else:
        write_json(output / "manifest.json", {"identity_hash": identity_hash, "configuration": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    records = []
    results_path = output / "episodes.jsonl"
    if results_path.exists():
        records = _read_episode_records(results_path,recover_final=resume)
    # Recorded terminal outcomes are immutable. Interrupted episodes without a row restart from reset.
    finished = {(row["task_id"],row["episode_index"]) for row in records}
    run_status, failure = "running", None
    started = time.monotonic()
    try:
        if client:
            client.preflight()
        with SimProcess(sim_config) as sim:
            sim_tasks = sim.list_tasks()
            if any(i < 0 or i >= len(sim_tasks) for i in task_ids):
                raise ValueError("requested task_id outside selected suite")
            simulator_manifest = {"metadata": sim.metadata, "tasks": sim_tasks}
            metadata_path = output / "simulator.json"
            if resume and metadata_path.exists():
                if json.loads(metadata_path.read_text()) != simulator_manifest:
                    raise ValueError("resume rejected: simulator source/controller provenance changed")
            write_json(metadata_path, simulator_manifest)
            protocol_hash = digest({"simulator":asdict(sim_config), "metadata":sim.metadata,
                                    "observation_source":identity["software"]["source_sha256"]["observation.py"]})
            for task_id in task_ids:
                for episode_index in episode_indices:
                    if (task_id,episode_index) in finished:
                        continue
                    episode_id = f"{sim_config.suite}_t{task_id:02d}_i{episode_index:02d}_{method}"
                    attempts = len(list(output.glob(episode_id + "_attempt*.jsonl.gz")))
                    log_path = output / f"{episode_id}_attempt{attempts:02d}.jsonl.gz"
                    row = {"suite":sim_config.suite,"seed":sim_config.seed,"method":method,
                           "task_id":task_id,"episode_index":episode_index,"episode_id":episode_id,
                           "protocol_hash":protocol_hash,"phase":phase,"status":"interrupted",
                           "steps":0,"path_length_m":0.0,"stage_changes":0,"planner_statuses":{},
                           "log":log_path.name, "attempt":attempts}
                    recorder = None
                    episode_started = time.monotonic()
                    try:
                        obs = sim.reset(task_id, episode_index)
                        policy.reset()
                        row["initialization"] = dict(sim.status)
                        row["task_prompt"] = obs.prompt
                        if save_video:
                            from .video import EpisodeVideoRecorder
                            video_name = episode_id+f"_attempt{attempts:02d}.mp4"
                            recorder = EpisodeVideoRecorder(output/video_name,
                                grid=getattr(policy, "grid", None), episode_id=episode_id)
                            recorder.observe(obs, step=0)
                        with gzip.open(log_path,"wt") as log:
                            log.write(json.dumps({"event":"reset", "observation":obs.to_dict(),
                                                  "sim_status":dict(sim.status)},allow_nan=False)+"\n")
                            log.flush()
                            steps = count() if sim_config.max_steps is None else range(sim_config.max_steps)
                            for step in steps:
                                if sim.status.get("success"):
                                    row["status"] = "success"
                                    break
                                if sim.status.get("terminated"):
                                    row["status"] = "timeout" if sim.status.get("horizon_reached") else "sim_terminated"
                                    break
                                write_json(state_path,{"status":"running","current_episode":episode_id,
                                    "step":step,"stage":"deciding",
                                    "budget":client.budget_snapshot() if client else None})
                                decision_started = time.monotonic()
                                decision = policy.act(obs)
                                decision_latency = time.monotonic()-decision_started
                                entry = {"step":step,"state":list(obs.state),"action":list(decision.action),
                                         "diagnostics":decision.diagnostics,
                                         "decision_latency_s":decision_latency}
                                if decision.diagnostics.get("refreshed"):
                                    entry["observation"] = obs.to_dict()
                                log.write(json.dumps(entry,allow_nan=False)+"\n")
                                log.flush()
                                if recorder:
                                    recorder.decision(obs, decision, decision_latency_s=decision_latency)
                                # Persist after each model/planner step; raw API logs cover crash gaps.
                                write_json(state_path,{"status":"running","current_episode":episode_id,
                                    "step":step,"stage":"executing",
                                    "budget":client.budget_snapshot() if client else None})
                                previous = np.asarray(obs.state[:3])
                                obs, success = sim.step(decision.action)
                                # Evaluator-only outcome, never included in the next policy request.
                                log.write(json.dumps({"step":step,"event":"executed",
                                    "next_state":list(obs.state),"success":success,
                                    "observation":obs.to_dict(),
                                    "sim_status":dict(sim.status)},allow_nan=False)+"\n")
                                log.flush()
                                row["steps"] += 1
                                row["path_length_m"] += float(np.linalg.norm(np.asarray(obs.state[:3])-previous))
                                status = decision.diagnostics["status"]
                                row["planner_statuses"][status] = row["planner_statuses"].get(status,0)+1
                                row["stage_changes"] += int(decision.diagnostics.get("stage_changed",False))
                                if recorder:
                                    recorder.executed(obs,step=step,success=success,sim_status=dict(sim.status))
                                if success:
                                    row["status"] = "success"
                                    break
                                if sim.status.get("terminated"):
                                    row["status"] = "timeout" if sim.status.get("horizon_reached") else "sim_terminated"
                                    break
                            else:
                                row["status"] = "timeout"
                    except KeyboardInterrupt:
                        row.update(status="interrupted",error="Run interrupted before task success")
                        run_status = "interrupted"
                    except BudgetExceeded as exc:
                        row.update(status="budget_exceeded",error=str(exc))
                        run_status = "budget_exceeded"
                    except SchemaError as exc:
                        row.update(status="schema_failure",error=str(exc))
                    except (APIError,MissingCredentials) as exc:
                        row.update(status="api_failure",error=str(exc))
                    except SimulatorError as exc:
                        row.update(status="sim_error",error=str(exc))
                    except (ValueError,TypeError,KeyError) as exc:
                        row.update(status="policy_error",error=f"{type(exc).__name__}: {exc}")
                    finally:
                        if recorder:
                            recorder.close(status=row["status"],error=row.get("error"))
                            row["video"] = video_name
                    row["elapsed_s"] = time.monotonic()-episode_started
                    row["budget_after"] = client.budget_snapshot() if client else None
                    records.append(row)
                    with results_path.open("a") as results:
                        results.write(json.dumps(row,allow_nan=False)+"\n")
                    write_json(output/"summary.json",summarize(records))
                    if run_status in ("budget_exceeded", "interrupted"):
                        break
                if run_status in ("budget_exceeded", "interrupted"):
                    break
            if run_status == "running":
                run_status = "complete"
    except BaseException as exc:
        run_status = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(state_path,{"status":run_status,"error":failure,
            "completed_episode_records":len(records),"requested_episodes":len(task_ids)*len(episode_indices),
            "elapsed_this_process_s":time.monotonic()-started,
            "budget":client.budget_snapshot() if client else None})
    return {"status":run_status,"output":str(output),"summary":summarize(records)}
