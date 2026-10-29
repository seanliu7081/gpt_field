"""Version A V1: fresh plan drafts only. No approval or task execution path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.update(MUJOCO_EGL_DEVICE_ID="3", MUJOCO_GL="egl", PYTHONPATH=str(ROOT / "src"),
                  PYTHONDONTWRITEBYTECODE="1")

import evaluate_agent_speedup_v2_user_check as libero
import evaluate_robocasa_user_check as coffee
from gpt_field.chatgpt_auth import SubscriptionCredentials
from gpt_field.agent_speedup_v2_user_check_transport import configure_client_transport
from gpt_field.experiment import write_json


def unchanged():
    frozen = json.loads((HERE / "uc71_release_manifest.json").read_text())
    for name, digest in frozen["runtime_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Release source changed: {name}")


def main():
    destination = HERE / "v1_planning"
    destination.mkdir(exist_ok=False)
    config = libero.base.configuration()
    token_provider = SubscriptionCredentials().access_token
    cases = [(f"libero_10_t{index}", f"libero_10_t{index}") for index in range(10)]
    cases.extend([("libero_goal_g5", "libero_goal_g5"),
                  ("libero_10_t6_relayout", "libero_10_t6"), ("robocasa_coffee", None)])
    results = {"version": libero.VERSION, "scope": "V1 planning only",
               "task_executions": 0, "approvals": 0,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    write_json(destination / "summary.json", results)
    for name, task_key in cases:
        unchanged()
        started = time.monotonic()
        row = {"name": name, "review_dir": str(destination / name), "status": "preparing"}
        results["cases"].append(row)
        write_json(destination / "summary.json", results)
        print(json.dumps({"event": "plan_only_start", "name": name}), flush=True)
        try:
            if task_key is not None:
                sim_config, monitor = libero.base.build_sim_config(task_key)
                expected = None
                if name.endswith("_relayout"):
                    from dataclasses import replace
                    layout = ROOT / "runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6"
                    sim_config = replace(sim_config, initial_state_override=str(layout / "initial_state.npy"))
                    expected = "ddb1ee8d611863cb002fb3012d35244d39c790a7204750945004695a51d4ab3c"
                libero._live_gate(config, sim_config, task_key, 0, expected)
                store = libero.prepare_review(destination / name,
                    config=config, sim_config=sim_config, task_key=task_key, episode_index=0,
                    monitor_options=monitor, token_provider=token_provider,
                    expected_initial_observation_sha256=expected, video=True, planning_video=True,
                    reasoning_effort="high", request_deadline_s=180., stream_progress=True)
                review = store.load()
            else:
                def bounded_client(*args, **kwargs):
                    client = coffee.RoboCasaPlanClient(*args, **kwargs)
                    configure_client_transport(client, request_deadline_s=180., stream_progress=True)
                    return client
                review = coffee.prepare(destination / name, config=config,
                    sim_config=coffee.RoboCasaConfig(), token_provider=token_provider,
                    client_factory=bounded_client)
            assert review["status"] == "draft" and review["approved"] is None
            assert review["preparation"]["policy_actions"] == 0
            budget = review["preparation"]["model_budget"]
            assert budget["billing"] == "chatgpt_subscription" and budget["api_key_calls"] == 0
            row.update(status="draft_prepared", policy_actions=0, validation=review["validation"],
                       model_budget=budget, initial_observation_sha256=review["initial_observation_sha256"],
                       **libero.direction_review_output(review))
        except Exception as exc:
            row.update(status="preparation_error", error_type=type(exc).__name__,
                       error=libero.base.redact_secrets(str(exc)), policy_actions=0)
        row["preparation_elapsed_s"] = time.monotonic() - started
        unchanged()
        write_json(destination / "summary.json", results)
        print(json.dumps({"event": "plan_only_finished", "name": name, "status": row["status"],
                          "elapsed_s": row["preparation_elapsed_s"], "error": row.get("error")}), flush=True)
    results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(destination / "summary.json", results)
    return 0 if all(row["status"] == "draft_prepared" for row in results["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
