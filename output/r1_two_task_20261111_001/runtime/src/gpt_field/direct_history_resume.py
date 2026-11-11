"""Strict, read-only reconstruction of one completed direct-history prefix.

Only a fresh, budget-stopped run at a full five-action boundary is supported.
Every request must have one validated response and known usage. Unsupported
partial actions, retries, missing usage, or later continuation chains fail closed;
their costs are never silently dropped. Verification uses recorded responses in
an offline transport, never calls OpenAI, and never starts a simulator.

Simulator state reconstruction remains the continuation runner's responsibility.
After that independent replay gate, ``restore_policy`` copies the verified
conversation and policy state without executing or billing any replayed action.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Callable

from .astra import ACCOUNTING_VERSION, AstraConfig, BudgetExceeded, _canonical, _strict_loads
from .direct_history import (
    CHUNK_SIZE, HISTORY_PROTOCOL, PROMPT_VERSION, TASK,
    DirectHistoryAstraClient, DirectHistoryPolicy,
)
from .experiment import digest
from .observation import PolicyObservation
from .schema import GridSpec


ROOT = Path(__file__).resolve().parents[2]
MODE = "live_astra_direct_history"
REQUIRED_SOURCE = frozenset({
    "__init__.py", "__main__.py", "action.py", "astra.py", "baselines.py", "cli.py",
    "diagnostics.py", "direct_history.py", "direct_history_video.py", "experiment.py",
    "field.py", "libero.py", "observation.py", "planner.py", "policy.py", "report.py",
    "schema.py", "video.py",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _same_budget(left: dict, right: dict, label: str) -> None:
    _require(isinstance(left, dict) and isinstance(right, dict), f"{label}: invalid budget")
    _require({k: v for k, v in left.items() if k != "elapsed_s"}
             == {k: v for k, v in right.items() if k != "elapsed_s"}, f"{label}: ledger mismatch")
    for budget in (left, right):
        elapsed = budget.get("elapsed_s")
        _require(not isinstance(elapsed, bool) and isinstance(elapsed, (int, float))
                 and math.isfinite(elapsed) and elapsed >= 0, f"{label}: invalid elapsed time")


def _policy_digest(policy: DirectHistoryPolicy) -> str:
    material = {"step": policy.step_index, "chunk": policy.chunk,
                "prediction": policy.prediction, "feedback": policy.latest_feedback,
                "progress": policy._progress,
                "expected_observation": (policy._expected_observation.to_dict()
                                         if policy._expected_observation is not None else None),
                "history_stats": policy.client.history_stats()}
    return hashlib.sha256(_canonical(material).encode()).hexdigest()


@dataclass(frozen=True)
class VerifiedResume:
    source: Path
    manifest: dict
    summary: dict
    source_hashes: dict[str, str]
    observations: tuple[PolicyObservation, ...] = field(repr=False)
    actions: tuple[tuple[float, ...], ...] = field(repr=False)
    sim_statuses: tuple[dict, ...] = field(repr=False)
    budget: dict
    config: AstraConfig
    history_digest: str
    next_request_key: str
    verification: dict
    _policy: DirectHistoryPolicy = field(repr=False, compare=False)

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def api_calls(self) -> int:
        return self.budget["api_calls"]

    @property
    def final_observation(self) -> PolicyObservation:
        return self.observations[-1]


def verify_source_hashes(resume: VerifiedResume) -> None:
    """Recheck parent artifacts after potentially lengthy simulator replay."""
    expected_api = {name for name in resume.source_hashes if name.startswith("api/")}
    current_api = {f"api/{path.name}" for path in (resume.source / "api").glob("*.json")}
    _require(current_api == expected_api, "Parent API record set changed after verification")
    for name, expected in resume.source_hashes.items():
        path = resume.source / name
        _require(path.is_file() and _sha(path) == expected, f"Parent artifact changed: {name}")
    _verify_code(resume.manifest["configuration"])


def _verify_code(identity: dict) -> None:
    sources = identity["software"]["source_sha256"]
    _require(isinstance(sources, dict) and REQUIRED_SOURCE <= set(sources), "Frozen source manifest is incomplete")
    for name, expected in sources.items():
        _require(Path(name).name == name and name.endswith(".py"), "Invalid frozen source path")
        path = ROOT / "src/gpt_field" / name
        _require(path.is_file() and _sha(path) == expected, f"Frozen policy source changed: {name}")
    runner = identity["runner"]
    _require(runner["path"] == "scripts/evaluate_bowl_direct_history.py", "Unsupported parent runner")
    _require(_sha(ROOT / runner["path"]) == runner["sha256"], "Frozen parent runner changed")


def load_resume(source: str | Path, *, max_cost_usd: float = 300) -> VerifiedResume:
    """Validate and reconstruct one fresh terminal run without network or writes.

    The caller must separately record the user's authorization for the new total
    cap. It includes all inherited usage. This helper accepts only a finite cap
    increase, not an implicit extra allowance or an unlimited configuration.
    """
    source = Path(source).expanduser().resolve()
    _require(source.is_dir(), "Parent run directory is missing")
    hashes = {}

    def read(name, *, lines=False):
        path = source / name
        _require(path.is_file(), f"Missing parent artifact: {name}")
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        if lines:
            _require(data.endswith(b"\n"), f"Unfinished JSONL artifact: {name}")
            return [_strict_loads(line.decode("utf-8")) for line in data.splitlines()]
        return _strict_loads(data.decode("utf-8"))

    manifest = read("manifest.json")
    identity = manifest["configuration"]
    _require(manifest.get("identity_hash") == digest(identity), "Parent manifest identity hash mismatch")
    _require(identity.get("kind") == "fresh_direct_history_baseline"
             and identity.get("mode") == MODE, "Only fresh live direct-history parents are supported")
    _require(identity.get("task") == TASK and identity.get("task_id") == 0
             and identity.get("episode_index") == 0, "Parent task identity mismatch")
    _require(identity.get("evaluation_scored") is True and identity.get("max_policy_steps") is None
             and identity.get("diagnostic_action_count") is None, "Parent is not the unbounded live baseline")
    policy_spec = identity["policy"]
    for key, value in {"class": "gpt_field.direct_history.DirectHistoryPolicy", "method": "direct_history",
                       "update_steps": 5, "recent_image_observations": 2,
                       "field_generation": False, "local_field_planning": False,
                       "success_feedback_to_policy": False}.items():
        _require(policy_spec.get(key) == value, f"Parent policy protocol mismatch: {key}")
    sim = identity["simulator"]
    for key, value in {"suite": "libero_spatial", "seed": 7, "max_steps": None,
                       "settle_steps": 10, "render_size": 256, "image_size": 224}.items():
        _require(sim.get(key) == value, f"Parent simulator protocol mismatch: {key}")
    _verify_code(identity)
    old_config = AstraConfig(**identity["api"])
    _require(old_config.model == "gpt-6-astra" and old_config.max_api_calls is None
             and old_config.max_wall_time_s is None and old_config.max_output_tokens == 8192
             and old_config.timeout_s == 180, "Unsupported API configuration")
    _require((old_config.input_price_per_million, old_config.cached_input_price_per_million,
              old_config.cache_write_price_per_million, old_config.output_price_per_million)
             == (10, 1, 12.5, 50), "Parent pricing configuration changed")
    _require(not isinstance(max_cost_usd, bool) and isinstance(max_cost_usd, (int, float))
             and math.isfinite(max_cost_usd) and old_config.max_cost_usd is not None
             and max_cost_usd > old_config.max_cost_usd, "New total cap must be a finite increase")
    config = replace(old_config, max_cost_usd=float(max_cost_usd))

    summary = read("summary.json")
    _require(read("runner_summary.json") == summary, "Parent summary files disagree")
    _require(read("episodes.jsonl", lines=True) == [summary], "Parent episode record differs from summary")
    state = read("run_state.json")
    _require(summary.get("status") == state.get("status") == "budget_exceeded"
             and state.get("stage") == "finished", "Parent must be terminal at its budget cap")
    _require(summary.get("pending_action_index") is None and state.get("pending_action_index") is None,
             "Parent has an unresolved action")
    _require(summary.get("mode") == state.get("mode") == MODE and summary.get("task") == TASK
             and summary.get("task_id") == 0 and summary.get("episode_index") == 0
             and summary.get("seed") == 7 and summary.get("suite") == "libero_spatial",
             "Parent summary identity mismatch")
    _require(summary.get("evaluation_completed") is False and summary.get("libero_success_observed") is False
             and summary.get("success_rate") is None and summary.get("libero_success") is None
             and summary.get("scored_episodes") == 0 and summary.get("incomplete_episodes") == 1,
             "Parent is already complete or has inconsistent scoring")
    steps = summary.get("steps")
    _require(type(steps) is int and steps > 0 and steps % CHUNK_SIZE == 0 and state.get("step") == steps,
             "Parent must end on a complete five-action boundary")
    calls = steps // CHUNK_SIZE
    _require(summary.get("validated_action_chunks") == summary.get("api_calls") == calls
             and summary.get("measured_feedback_count") == steps and summary.get("mock_response_calls") == 0,
             "Parent has missing, retried, or unexecuted action chunks")
    budget = summary["budget"]
    _require(budget.get("unknown_usage_calls") == 0 and budget.get("budget_bound_violated") is False,
             "Unresolved usage requires separate conservative reconciliation; automatic resume is refused")
    _same_budget(budget, state["budget"], "Terminal run state")
    _require(budget.get("configuration") == asdict(old_config) and budget.get("api_calls") == calls,
             "Parent saved budget configuration differs")
    _require(summary.get("max_cost_usd") == old_config.max_cost_usd, "Parent summary cap differs")
    read("simulator.json")
    for optional in ("authorization.json", "episode.mp4", "episode.events.jsonl"):
        if (source / optional).is_file():
            hashes[optional] = _sha(source / optional)

    trace = read("trace.jsonl", lines=True)
    _require(len(trace) == 1 + 3 * steps and trace[0].get("event") == "reset", "Parent trace sequence is incomplete")
    _require(trace[0].get("mode") == MODE, "Parent trace mode differs")
    obs = PolicyObservation.from_dict(trace[0]["observation"])
    _require(obs.prompt == TASK, "Parent reset task differs")
    observations, actions, statuses = [obs], [], [trace[0]["sim_status"]]
    _require(statuses[0].get("steps") == 0 and statuses[0].get("success") is False
             and statuses[0].get("terminated") is False, "Parent reset is not an active initial state")

    paths = sorted((source / "api").glob("*.request.json"))
    _require(len(paths) == calls, "Parent request count differs; pending or extra attempts are unsupported")
    expected_api_names = set()
    for ordinal, path in enumerate(paths, 1):
        _require(path.name.startswith(f"{ordinal:06d}-"), "Parent API ordinals are missing or duplicated")
        expected_api_names.update(path.name.replace(".request.json", suffix)
                                  for suffix in (".request.json", ".response.json", ".result.json"))
    _require({path.name for path in (source / "api").glob("*.json")} == expected_api_names,
             "Parent API records are missing or contain unmatched attempts")
    cursor = 0
    current_result = None
    current_request_path = None
    elapsed = 0.0

    def offline_transport(endpoint, payload, headers, timeout_s):
        nonlocal cursor, current_result, current_request_path, elapsed
        _require(cursor < len(paths), "Replay attempted an unrecorded model request")
        path = paths[cursor]
        prefix = path.name.removesuffix(".request.json")
        request = read(f"api/{prefix}.request.json")
        response = read(f"api/{prefix}.response.json")
        current_result = read(f"api/{prefix}.result.json")
        current_request_path = path
        _require(request.get("request_ordinal") == cursor + 1 and request.get("attempt") == 1,
                 "Schema retry or request ordinal mismatch is unsupported")
        _require(request.get("prompt_version") == PROMPT_VERSION and request.get("history_protocol") == HISTORY_PROTOCOL
                 and request.get("image_history") == 2 and request.get("mode") == "direct"
                 and request.get("grid") == asdict(GridSpec())
                 and request.get("accounting_version") == ACCOUNTING_VERSION,
                 "Parent API protocol metadata differs")
        _require(request.get("configuration") == asdict(old_config), "Parent request configuration differs")
        _require(request.get("history_stats_before_request") == client.history_stats(), "Parent history counters differ")
        _require(request.get("payload") == payload, f"Parent request {cursor + 1} conversation does not reconstruct exactly")
        _require(request.get("request_key") == client.request_key(payload, GridSpec(), "direct"),
                 "Parent request key does not match its reconstructed payload")
        _require(request.get("reservation_usd") == client.budget_snapshot()["reservation_per_call_usd"]
                 and request.get("timeout_s") == timeout_s == old_config.timeout_s,
                 "Parent request budget reservation or timeout differs")
        timing = request.get("budget_elapsed_s_before")
        _require(not isinstance(timing, bool) and isinstance(timing, (int, float)) and math.isfinite(timing)
                 and timing >= elapsed, "Parent request timing is invalid or unordered")
        _require(current_result.get("status") == "valid", "Failed parent requests require separate reconciliation")
        next_elapsed = current_result.get("budget", {}).get("elapsed_s")
        _require(not isinstance(next_elapsed, bool) and isinstance(next_elapsed, (int, float))
                 and math.isfinite(next_elapsed) and next_elapsed >= timing, "Parent result timing is invalid")
        elapsed = next_elapsed
        usage = response.get("usage")
        _require(isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
                 for key in ("input_tokens", "output_tokens")), "Unknown parent usage cannot be dropped")
        _require(not response.get("offline_mock"), "Mock model responses cannot continue a live evaluation")
        cursor += 1
        return response

    client = DirectHistoryAstraClient(old_config, api_key="OFFLINE-RECONSTRUCTION-ONLY",
                                     transport=offline_transport, clock=lambda: 0.0)
    policy = DirectHistoryPolicy(client)
    _require(policy_spec.get("initial_history_settings") == client.history_stats(), "Initial history settings differ")
    path_length = 0.0
    gripper = {"close": 0, "open": 0, "neutral": 0}
    for step in range(steps):
        action_row, executed, feedback_row = trace[1 + 3 * step:4 + 3 * step]
        _require([row.get("event") for row in (action_row, executed, feedback_row)] == ["action", "executed", "feedback"]
                 and all(row.get("step") == step for row in (action_row, executed, feedback_row)),
                 f"Parent trace ordering mismatch at step {step}")
        _require(action_row.get("mode") == executed.get("mode") == MODE and action_row.get("state") == list(obs.state),
                 f"Parent action observation differs at step {step}")
        decision = policy.act(obs)
        _require(action_row.get("action") == executed.get("action") == list(decision.action),
                 f"Parent action does not match the model chunk at step {step}")
        ignored = {"api_latency_s", "api_request_log"}
        _require({k: v for k, v in action_row["diagnostics"].items() if k not in ignored}
                 == {k: v for k, v in decision.diagnostics.items() if k not in ignored},
                 f"Parent decision/history diagnostics differ at step {step}")
        if step % CHUNK_SIZE == 0:
            _require(Path(action_row["diagnostics"]["api_request_log"]).name == current_request_path.name,
                     "Parent action links the wrong API request")
            _same_budget(current_result["budget"], client.budget_snapshot(), f"Request {cursor}")
        after = PolicyObservation.from_dict(executed["observation"])
        _require(after.prompt == TASK and executed.get("next_state") == list(after.state), "Parent after-state differs")
        sim_status = executed["sim_status"]
        _require(executed.get("success") is False and sim_status.get("success") is False
                 and sim_status.get("terminated") is False and sim_status.get("steps") == step + 1,
                 "Parent contains successful, terminated, or unsequenced simulator steps")
        actual_feedback = policy.observe_transition(obs, decision, after)
        _require(actual_feedback == feedback_row.get("feedback"), f"Parent measured feedback differs at step {step}")
        path_length += actual_feedback["measured_translation_norm_m"]
        gripper["close" if decision.action[6] > 0 else "open" if decision.action[6] < 0 else "neutral"] += 1
        actions.append(decision.action)
        observations.append(after)
        statuses.append(sim_status)
        obs = after

    _require(cursor == calls and policy.step_index == steps and not policy.chunk and policy._pending is None,
             "Reconstruction did not end at the exact complete action boundary")
    _same_budget(budget, client.budget_snapshot(), "Reconstructed terminal")
    _require(budget["elapsed_s"] >= elapsed, "Terminal budget predates the final request")
    for name, count in gripper.items():
        _require(summary.get(f"gripper_{name}_actions") == count, "Parent gripper summary differs")
    _require(math.isclose(summary["path_length_m"], path_length, rel_tol=1e-12, abs_tol=1e-12),
             "Parent motion summary differs")
    try:
        client._reserve()
    except BudgetExceeded:
        pass
    else:
        raise ValueError("Parent saved ledger does not explain its budget stop")

    restored_budget = copy.deepcopy(budget)
    restored_budget["configuration"] = asdict(config)
    restored_budget["max_cost_usd"] = config.max_cost_usd
    _require(restored_budget["cost_usd_upper_bound"] + restored_budget["reservation_per_call_usd"]
             <= config.max_cost_usd, "New total cap cannot reserve the next request")
    history_digest = hashlib.sha256(_canonical(client.history_snapshot()).encode()).hexdigest()
    next_payload = client.build_payload(obs.to_dict(), GridSpec(), "direct")
    next_key = client.request_key(next_payload, GridSpec(), "direct")
    verification = {"status": "verified", "parent_identity_hash": manifest["identity_hash"],
                    "steps": steps, "state_observations": len(observations), "feedback_records": steps,
                    "validated_requests": calls, "next_global_request_ordinal": calls + 1,
                    "next_request_key": next_key, "history_sha256": history_digest,
                    "policy_state_sha256": _policy_digest(policy),
                    "history_stats": client.history_stats(), "model_api_calls": 0,
                    "simulator_started": False, "policy_changed": False,
                    "previous_max_cost_usd": old_config.max_cost_usd,
                    "new_cumulative_max_cost_usd": config.max_cost_usd,
                    "inherited_cost_usd_upper_bound": budget["cost_usd_upper_bound"],
                    "unresolved_usage_calls": 0}
    resume = VerifiedResume(source, manifest, summary, hashes, tuple(observations), tuple(actions), tuple(statuses),
                            restored_budget, config, history_digest, next_key, verification, policy)
    verify_source_hashes(resume)
    return resume


def restore_policy(resume: VerifiedResume, *, api_key: str | None,
                   transport: Callable | None = None, log_dir: str | Path | None = None) -> DirectHistoryPolicy:
    """Restore the exact next decision state, with inherited global API ordinals.

    ``log_dir`` must be new or empty. The inherited generic reconciliation method
    assumes logs begin at ordinal 1; it must NOT be used on this new segment's
    suffix alone. Future interruption recovery must reconcile the complete chain.
    """
    verify_source_hashes(resume)
    if log_dir is not None:
        path = Path(log_dir)
        _require(not path.exists() or (path.is_dir() and not any(path.iterdir())), "Continuation API log directory is not empty")
    original = resume._policy
    _require(original.step_index == resume.steps and not original.chunk and original._pending is None,
             "Verified policy state was modified after reconstruction")
    _require(hashlib.sha256(_canonical(original.client.history_snapshot()).encode()).hexdigest() == resume.history_digest,
             "Verified conversation was modified after reconstruction")
    _require(_policy_digest(original) == resume.verification["policy_state_sha256"],
             "Verified policy feedback state was modified after reconstruction")
    # Nested snapshot mappings are inspectable by the caller. Re-derive the sole
    # authorized change from the still-hashed source before trusting those maps.
    original_summary = _strict_loads((resume.source / "summary.json").read_text())
    expected_budget = copy.deepcopy(original_summary["budget"])
    expected_config = dict(expected_budget["configuration"])
    expected_config["max_cost_usd"] = resume.config.max_cost_usd
    _require(asdict(resume.config) == expected_config, "Restored configuration changes more than the cost cap")
    expected_budget["configuration"] = expected_config
    expected_budget["max_cost_usd"] = resume.config.max_cost_usd
    _require(resume.budget == expected_budget, "Verified inherited budget was modified after reconstruction")
    client = DirectHistoryAstraClient(resume.config, api_key=api_key, transport=transport, log_dir=log_dir)
    policy = DirectHistoryPolicy(client)
    client.load_budget_snapshot(copy.deepcopy(resume.budget))
    client._history = original.client.history_snapshot()
    client._observation_count = original.client._observation_count
    client._feedback_count = original.client._feedback_count
    policy.step_index = original.step_index
    policy.prediction = copy.deepcopy(original.prediction)
    policy.latest_feedback = copy.deepcopy(original.latest_feedback)
    policy._expected_observation = resume.final_observation
    policy._progress = copy.deepcopy(original._progress)
    payload = client.build_payload(resume.final_observation.to_dict(), GridSpec(), "direct")
    _require(client.request_key(payload, GridSpec(), "direct") == resume.next_request_key,
             "Restored next-request payload differs from verified continuation")
    return policy
