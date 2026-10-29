"""Evaluate agent_SpeedUp_v2: model-planned relations executed by a local, sensor-grounded relation layer.

One fresh LIBERO episode per process for a v2 registry key (libero_10 task 2 is the primary task; bowl_plate =
LIBERO-Spatial 0 and drawer_bowl = LIBERO-Goal 3 are the v1 regressions) at native 512 px, seed 7, settle 10,
state8, no horizon, under v1's speed-first eval protocol: contacts are recorded only (generic name-free monitor,
plus v1's monitor chained on its own two tasks), raw LIBERO success is the only completion (stop_on_success),
an operator stop file ends the run at a complete-feedback boundary, a fresh output directory is required, and
the manifest freezes the identity, the source and the user-modified helper chain.

At finalization, after the terminal status is fixed and before the simulator closes, a diagnosis-only strict
physical truth check runs in the worker (terminal state, then the release-and-settle counterfactual, then the
BDDL answer key and the plan parse check). It never changes status, task_success or task_evaluation.json;
summary.physical_truth reports AC-2's verdict (physical_success / lenient_success / raw_failure) and
summary.v2_success = raw LIBERO success AND strict physical truth (the same rule in live and smoke mode, lead
decision L5). A truth failure is a cleanup_error that never promotes the status to infrastructure_error. Truth is
skipped when the run was interrupted inside a simulator request (the JSONL protocol has no request ids, so the
stale reply would be read as the truth reply), and every truth reply's shape is checked against the op sent.

Modes:
- --preflight: a synthetic 512 observation; prints the identity, the official prompt the client is constructed
  with and its acceptance by the v2 client's prompt check, the constants, instruction, schema and lint hashes, the
  robot facts and the real v2 plan / point_fine / point_coarse / repair payloads (images as placeholders). No
  credentials, network, simulator or files.
- --sim-smoke --smoke-script plumbing: the real simulator, zero model calls, unscored, with a robot-only scripted
  hold policy (empty close/open, small lifts and yaws; no object interaction).
- --sim-smoke --smoke-script relation: the real simulator and the real v2 policy, client and relation layer, with
  the ORACLE stub (scripts/agent_speedup_v2_oracle.py, evaluator geometry) answering the client's requests through
  ScriptedV2Transport; zero model calls, unscored. Only this branch starts the worker with the oracle op enabled.
- --live: one scored episode with the ChatGPT subscription (gpt-6-astra, medium effort): the v2 client with the
  ordered SSE transport and the subscription credentials (--store-dir / --profile, as v1), and the default v2
  policy AgentSpeedUpV2Policy(client) with no overrides. The ORACLE is refused (worker option and runner branch).
  User decision 2026-10-19 (live-first): there is no zero-model-call gate before a live run; a live run refuses to
  start when the T0 lint (require_all) fails or the zero-call build check fails (the real payloads built and the
  worked example parsed against the plan schema; recorded as manifest live_build_check). --stop-after-steps N
  is an operator stop at the first complete-feedback boundary after N executed actions (status user_stop).

The stuck alert (operator alert only, no automatic cap): >= 10 of the last 12 relation outcomes failed or were
rejected; a relation (per relation and part ids) once met failed to re-establish more than 5 times; an
invalid_plan rejection (at once). While an alert stands (no success since), every 12 further failed or rejected
outcomes alert again. Registration and calibration outcomes (a repoint's re-registration) neither count nor clear
an alert. Each alert prints the operator stop command.

Recovery exhausted (gap round, ruling L3): when the policy emits the v2 event ``recovery_exhausted`` (its repair
ladder skipped a step and nothing can run any more; it then holds with zero calls) the runner raises a stuck alert
with ``automatic_cap: True`` and ends the episode at the next complete-feedback boundary with the failure status
``recovery_exhausted`` (never a success; terminal truth is still read, diagnosis only). Without it the run would
idle to the cap.

Trace: each v2 event (diagnostics["v2_events"]) is one row under its own name (its only copy; the action row
carries the count as v2_event_rows); the act's pointing results (diagnostics["pointing"]) are "pointing_result"
rows, distinct from the relation layer's v2 "pointing" event.

scripts/evaluate_agent_speedup_v1.py is loaded read-only for its protocol helpers; nothing of v1 is edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = "scripts/evaluate_agent_speedup_v2.py"
V1_RUNNER_PATH = "scripts/evaluate_agent_speedup_v1.py"
ORACLE_PATH = "scripts/agent_speedup_v2_oracle.py"
_spec = importlib.util.spec_from_file_location("_agent_speedup_v1_runner_readonly", ROOT / V1_RUNNER_PATH)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

import gpt_field.experiment as _experiment  # noqa: E402
from gpt_field import agent_speedup_v2_robot as robot  # noqa: E402
from gpt_field import agent_speedup_v2_thresholds as thresholds  # noqa: E402
from gpt_field.agent_speedup_v2_contacts import FORBIDDEN, MODE as CONTACT_MODE  # noqa: E402
from gpt_field.agent_speedup_v2_sim import V2SimProcess, initial_observation_sha256  # noqa: E402
from gpt_field.agent_speedup_v2_tasks import (  # noqa: E402
    DEFAULT_LIBERO_ROOT, DEFAULT_SIM_PYTHON, IMAGE_SIZE, PRIMARY_TASK, V2_TASK_ALIASES, V2_TASK_KEYS,
    build_v2_sim_config, collision_base_mode, frozen_reset_sha256, resolve_v2_task)
from gpt_field.agent_speedup_v2_truth import parse_check, physical_summary  # noqa: E402
from gpt_field.policy import Decision  # noqa: E402

# v1 protocol helpers (read-only).
configuration = _v1.configuration
digest, write_json, source_manifest = _v1.digest, _v1.write_json, _v1.source_manifest
_validate_task_configuration = _v1._validate_task_configuration
_validate_episode_index = _v1._validate_episode_index
termination_as_interrupt = _v1.termination_as_interrupt
_simulator_without_credentials = _v1._simulator_without_credentials
_safe_error, _append = _v1._safe_error, _v1._append
_FinalizationSignals, NullRecorder = _v1._FinalizationSignals, _v1.NullRecorder
redact_secrets, _write_traceback, _error_record = _v1.redact_secrets, _v1._write_traceback, _v1._error_record
check_render_environment, render_environment = _v1.check_render_environment, _v1.render_environment
_checked_json, _validate_expected_sha, _sha256 = _v1._checked_json, _v1._validate_expected_sha, _v1._sha256
_payload_preview = _v1._payload_preview
APIError, MissingCredentials, BudgetExceeded, SchemaError = (_v1.APIError, _v1.MissingCredentials,
                                                             _v1.BudgetExceeded, _v1.SchemaError)
PolicyObservation, image_to_data_url = _v1.PolicyObservation, _v1.image_to_data_url
OPERATOR_STOP_FILE, TRACEBACK_FILE = _v1.OPERATOR_STOP_FILE, _v1.TRACEBACK_FILE
FINALIZATION_ABORT_PRESSES = _v1.FINALIZATION_ABORT_PRESSES
V1_HELPER_SCRIPTS = tuple(_v1.HELPER_SCRIPTS)

METHOD = "agent_speedup_v2"
RUNNER_VERSION = "agent-speedup-v2-runner-m6-r4"   # r2: live build check, stuck re-alerts, trace rows (BI review)
# r3 (integration, live smokes): registration / calibration outcomes do not count in the stuck alert
# r4 (gap round, ruling L3): recovery_exhausted -> automatic_cap stuck alert, status recovery_exhausted
LIVE_MODE = "live_chatgpt_subscription_agent_speedup_v2"
SMOKE_MODE = "offline_scripted_agent_speedup_v2_sim_smoke"
CONTROLLER = "agent_relation"
SMOKE_SCRIPTS = ("plumbing", "relation")
DEFAULT_SMOKE_STEPS = 150
DEFAULT_RELATION_SMOKE_STEPS = 1300          # the AC-4 step ceiling (GS-6 runs the chain with --smoke-steps 1300)
MAX_SMOKE_STEPS = 2000
EXECUTED_CONSOLE_EVERY = 25
STUCK_WINDOW, STUCK_FAILURES = 12, 10
STUCK_REESTABLISH_FAILURES = 5               # spec §4.11 / B5: a met relation failing to re-establish > 5 times
# While an alert stands (no success since), every further STUCK_REPEAT_FAILURES failed / rejected outcomes alert
# again (operator alert only, no cap): a repair loop keeps announcing itself instead of one alert then silence.
STUCK_REPEAT_FAILURES = STUCK_WINDOW
# Bookkeeping outcomes that are no progress of a plan step (integration, live smokes 2026-10-19): every repair
# "repoint" records a succeeded "registered" outcome, so a handle re-point loop (held failed, supported_on failed,
# registered succeeded, ...) never reached 10 of 12 and each registration re-armed / cleared the alert. They are
# printed and recorded as before but neither enter the stuck window nor clear a standing alert.
STUCK_IGNORED_RELATIONS = ("registered", "calibrated")
# The short first live run's operator stop (printed by --preflight as live_command_first_run; not a protocol cap).
LIVE_FIRST_RUN_STOP_AFTER_STEPS = 300
# recovery_exhausted (ruling L3): the policy's repair ladder gave up; a clean terminal failure, never a success
RECOVERY_EXHAUSTED_EVENT = RECOVERY_EXHAUSTED_STATUS = "recovery_exhausted"
OUTCOME_STATUSES = ("success", "user_stop", "diagnostic_complete", "diagnostic_sim_terminated",
                    RECOVERY_EXHAUSTED_STATUS)
TRUTH_STATUSES = (*OUTCOME_STATUSES, "subscription_error")
# The terminal statuses whose raw LIBERO success can make a v2_success (live: success; smoke: the simulator ended
# the smoke on raw success). Lead decision L5: the same raw-AND-strict rule in both modes.
V2_SUCCESS_STATUSES = ("success", "diagnostic_sim_terminated")
# Diagnosis-only finalization reads: their failures are cleanup_errors that never change the status (spec §6.5).
TRUTH_READS = ("truth_terminal", "truth_after_release_settle", "answer_key")
TRUTH_STAGE = "finalizing_truth"
V1_CHAINED_MODE = "task_manipulation_v1"
STOPPING_RULE = ["LIBERO success", "operator stop file", "user stop",
                 "subscription authorization or quota error", "infrastructure error",
                 "recovery_exhausted (the policy's repair ladder ended; automatic cap, failure status)"]
LIVE_GATE = ("none: user decision 2026-10-19 (live-first); no zero-model-call gate before live runs; the T0 lint "
             "(require_all) must pass for a CLI live run")
PREFLIGHT_STATE = (0.0, 0.0, 1.0, math.pi, 0.0, 0.0, 0.04, -0.04)
# The user-modified helper chain (spec §6.1, critic COULD 2): its sha256 is part of the identity.
HELPER_CHAIN = ("scripts/evaluate_bowl_hybrid_subscription.py", "scripts/evaluate_bowl_subscription_close.py",
                "scripts/evaluate_bowl_field_history.py", "src/gpt_field/hybrid_subscription.py",
                "src/gpt_field/direct_history.py", "src/gpt_field/chatgpt_plan.py", "src/gpt_field/observation.py",
                "src/gpt_field/policy.py", "src/gpt_field/libero.py")
DATA_FILES = ("src/gpt_field/robot_facts_v2.json",)
# v2 trace events echoed to the console (compact scalar fields only) so an operator can follow a live run.
CONSOLE_EVENTS = ("relation_plan", "plan_validation", "step_start", "step_end", "repair", "review", "model_claim",
                  "invariant", "stays_check", RECOVERY_EXHAUSTED_EVENT)
CONSOLE_FIELDS = 10
CONSOLE_TEXT_CHARS = 120


# ---------------------------------------------------------------------------------------------------------
# validation
def _validate_smoke_steps(value):
    if type(value) is not int or not 1 <= value <= MAX_SMOKE_STEPS:
        raise ValueError(f"smoke_steps must be an integer in [1, {MAX_SMOKE_STEPS}]")
    return value


def _validate_stop_after_steps(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("stop_after_steps must be a positive integer (an operator stop) or None")
    return value


def build_sim_config(task_key, *, sim_python=DEFAULT_SIM_PYTHON, libero_root=DEFAULT_LIBERO_ROOT):
    return build_v2_sim_config(task_key, sim_python=sim_python, libero_root=libero_root)


def _validate_collision_configuration_v2(sim_config, task_key):
    """v2 copy of the v1 validator: the base worker mode is 'none' (or v1's own mode on v1's own tasks, chained
    under the generic monitor), and contacts are record-only."""
    mode = getattr(sim_config, "collision_check_mode", None)
    if mode != collision_base_mode(task_key):
        raise ValueError("v2 requires the registry's base collision mode with the generic_contact_v2 monitor")
    if getattr(sim_config, "stop_on_collision", True):
        raise ValueError("v2 contacts are record-only; stop_on_collision must be False")


def validate_configuration(config, sim_config, task_spec):
    _validate_task_configuration(config, sim_config, task_spec, speed_first_v3=True)
    _validate_collision_configuration_v2(sim_config, task_spec.key)
    if sim_config.include_arm_joints:
        raise ValueError("agent_speedup_v2 uses the state8 protocol; include_arm_joints must be False")
    if sim_config.initial_state_override is not None or sim_config.task_bddl_override is not None:
        raise ValueError("agent_speedup_v2 evaluates official LIBERO scenes only; overrides are unsupported")
    if not sim_config.stop_on_success:
        raise ValueError("LIBERO success must stop the episode")
    if sim_config.max_steps is not None or sim_config.seed != 7 or sim_config.settle_steps != 10:
        raise ValueError("agent_speedup_v2 uses seed 7, settle 10 and no action horizon")


def _collision_record_v2(sim_status):
    """v1's _collision_record invariants for the generic record-only monitor."""
    record = sim_status.get("collision")
    if not isinstance(record, dict) or record.get("enabled") is not True or record.get("mode") != CONTACT_MODE:
        raise ValueError("Required generic_contact_v2 monitor is missing or disabled")
    if type(record.get("detected")) is not bool or type(sim_status.get("libero_success")) is not bool:
        raise ValueError("Collision monitor and raw LIBERO status must be explicit")
    if record.get("stop_on_collision") is not False or record.get("collision_policy") != "record_only":
        raise ValueError("v2 contact monitor must be record-only")
    if sim_status.get("success") != sim_status["libero_success"]:
        raise ValueError("v2 simulator success must equal raw LIBERO success")
    if record["detected"] and not sim_status["libero_success"] and sim_status.get("terminated"):
        raise ValueError("A recorded contact must not terminate a v2 episode")
    return dict(record)


def _collision_v1_record(sim_status, chained):
    """The chained v1 monitor record (v1's _collision_record invariants): present, enabled, v1's own mode and
    record-only when chained; absent otherwise."""
    record = sim_status.get("collision_v1_record_only")
    if not chained:
        if record is not None:
            raise ValueError("A v1 monitor record appeared although the v1 monitor is not chained")
        return None
    if (not isinstance(record, dict) or record.get("enabled") is not True or record.get("mode") != V1_CHAINED_MODE
            or record.get("stop_on_collision") is not False or record.get("collision_policy") != "record_only"):
        raise ValueError("chained v1 monitor must be record-only")
    return dict(record)


def _truth_reply_shape_ok(label, value):
    """The reply matches the op that was sent (a misaligned JSONL reply has another shape)."""
    if not isinstance(value, dict):
        return False
    if label == "truth_terminal":
        return value.get("state") == "terminal" and isinstance(value.get("atoms"), list)
    if label == "truth_after_release_settle":
        return value.get("state") == "after_release_settle" and isinstance(value.get("atoms"), list)
    if label == "answer_key":
        return isinstance(value.get("goal_atoms"), list)
    raise ValueError(label)


def _client_identity(client, config):
    """v1's _client_identity fields (scripts/evaluate_agent_speedup_v1.py) without v1's target description."""
    stats = client.history_stats() if callable(getattr(client, "history_stats", None)) else {}
    identity = {"class": type(client).__qualname__, "model": getattr(config, "model", None),
                "prompt_version": getattr(client, "prompt_version", None),
                "reasoning_effort": getattr(client, "reasoning_effort", None),
                "history_protocol": (stats or {}).get("protocol"), "initial_history_settings": stats,
                "task_prompt": getattr(client, "task_prompt", None)}
    if hasattr(client, "offline_mock") and not isinstance(client, OfflinePlumbingClient):
        transport = getattr(client, "transport", None)
        identity.update(offline_mock=getattr(client, "offline_mock", False) is True,
                        transport=getattr(transport, "__qualname__", None) or type(transport).__qualname__)
    return identity


def _policy_identity(policy):
    """The constructed policy: class, controller, versions and the (always empty) overrides (§C13.1)."""
    module = sys.modules.get(type(policy).__module__)
    layer = getattr(policy, "_layer", None)
    return {"class": type(policy).__qualname__, "module": type(policy).__module__,
            "controller": getattr(policy, "controller", None),
            "policy_version": getattr(module, "POLICY_VERSION", None),
            "relations_version": getattr(layer, "RELATIONS_VERSION", None),
            "overrides": dict(getattr(policy, "_overrides", None) or {}),
            "episode_factory": None if getattr(policy, "_episode_factory", None) is None else "injected",
            "task_key_reaches_policy": False}


def _relation_key(item):
    pids = item.get("pids")
    return f"{item.get('relation') or item.get('skill')}:{json.dumps(pids, sort_keys=True, default=str)}"


def _stuck_counted(item) -> bool:
    """A relation outcome the stuck alert counts: not a STUCK_IGNORED_RELATIONS bookkeeping outcome."""
    return (item.get("relation") or item.get("skill")) not in STUCK_IGNORED_RELATIONS


def _stuck_reason(outcomes):
    """Operator alert only (no automatic cap), spec §4.11 / §3.4 (B5): a relation once met (per relation and part
    ids) failed to re-establish more than STUCK_REESTABLISH_FAILURES times; >= 10 of the last 12 relation outcomes
    failed or were rejected. Registration and calibration outcomes are not counted (STUCK_IGNORED_RELATIONS)."""
    outcomes = [item for item in outcomes if _stuck_counted(item)]
    last = outcomes[-1] if outcomes else None
    if last is not None and last.get("status") == "failed":
        key, met, failures = _relation_key(last), False, 0
        for item in outcomes:
            if _relation_key(item) != key:
                continue
            if item.get("status") == "succeeded":
                met = True
            elif met and item.get("status") == "failed":
                failures += 1
        if failures > STUCK_REESTABLISH_FAILURES:
            return (f"relation {key} was met once and has failed to re-establish {failures} times "
                    f"(> {STUCK_REESTABLISH_FAILURES})")
    window = outcomes[-STUCK_WINDOW:]
    failing = sum(item.get("status") in ("failed", "rejected") for item in window)
    if failing >= STUCK_FAILURES:
        return f"{failing} of the last {len(window)} relation outcomes failed or were rejected"
    return None


def _immediate_stuck_reason(record):
    """An invalid_plan rejection raises the alert at once, armed or not (spec §3.4)."""
    if record.get("status") == "rejected" and record.get("reason") == "invalid_plan":
        return "the relation plan was rejected twice (invalid_plan); the policy holds still"
    return None


def _repeat_stuck_reason(failures_since_alert, alert):
    """While an alert stands (no succeeded outcome since), every STUCK_REPEAT_FAILURES further failed / rejected
    outcomes alert again (operator alert only, no automatic cap)."""
    if alert is None or failures_since_alert < STUCK_REPEAT_FAILURES:
        return None
    return (f"still stuck: {failures_since_alert} further relation outcomes failed or were rejected since the alert "
            f"at step {alert.get('step')}, with no success in between")


# Plan relations against the BDDL answer key (contract §C14.4's mapping, kept runner-side so the M0 truth module and
# its recorded real-sim truth probes stay byte-identical): turn + an until.effect criterion <-> TurnOn / TurnOff;
# place_on <-> On; place_in <-> In, or On(object); slide <-> Open / Close; push_to <-> On(zone). Diagnosis only.
PLAN_DO_FOR_PREDICATE = {"turnon": ("turn",), "turnoff": ("turn",), "open": ("slide", "turn"),
                         "close": ("slide", "turn"), "on": ("place_on", "push_to", "place_in"), "in": ("place_in",)}
PLAN_MATCH_VERSION = "agent-speedup-v2-plan-match-1"
_NAME_STOPWORDS = frozenset({"region", "init", "contain", "top", "side", "main", "cook", "the", "of", "on", "in"})


def _name_tokens(text):
    import re
    return {token for token in re.split(r"[^a-z]+", str(text or "").lower()) if len(token) > 1} - _NAME_STOPWORDS


def _plan_step_matches(atom, step):
    """Relation-level match of one plan step to one BDDL atom (no object names)."""
    predicate = str(atom[0]).lower()
    do = step.get("do")
    if do not in PLAN_DO_FOR_PREDICATE.get(predicate, ()):
        return False
    until = [item.get("o") for item in step.get("until") or [] if isinstance(item, dict)]
    if predicate in ("turnon", "turnoff"):
        return "effect" in until
    if predicate in ("open", "close") and do == "slide":
        return step.get("sense") in (predicate, "any", None)
    return True


def _plan_text_match(atom, step, texts):
    """Does each atom argument share a name token with the part text of the step's obj (and ref)?"""
    args = [str(item) for item in atom[1:]]
    pids = [step.get("obj")] if len(args) == 1 else [step.get("obj"), step.get("ref")][:len(args)]
    checks = [bool(_name_tokens(texts.get(pid, "")) & _name_tokens(arg)) for arg, pid in zip(args, pids)]
    return bool(checks) and all(checks)


def plan_atom_check(answer_key, plan):
    """``planned`` true / false per BDDL goal atom for the validated plan JSON (None without a plan)."""
    atoms = (answer_key or {}).get("goal_atoms") or []
    if not isinstance(plan, dict):
        return {"plan_available": False, "atoms": [{"atom": atom, "planned": None} for atom in atoms],
                "version": PLAN_MATCH_VERSION}
    steps = [step for step in plan.get("steps") or [] if isinstance(step, dict)]
    texts = {part.get("id"): str(part.get("text") or "") for part in plan.get("parts") or []
             if isinstance(part, dict)}
    rows, used = [], set()
    for atom in atoms:
        matched = [i for i, step in enumerate(steps) if atom and _plan_step_matches(atom, step)]
        named = [i for i in matched if _plan_text_match(atom, steps[i], texts)]
        used.update(matched)
        rows.append({"atom": atom, "planned": bool(matched), "plan_steps": matched,
                     "plan_steps_text_match": named, "planned_with_text_match": bool(named)})
    return {"plan_available": True, "atoms": rows,
            "all_atoms_planned": all(row["planned"] for row in rows) if rows else None,
            "plan_steps_without_atom": [i for i in range(len(steps)) if i not in used],
            "hold_true": [dict(item) for item in plan.get("hold_true") or [] if isinstance(item, dict)],
            "rule": "relation level: turn+effect / TurnOn-TurnOff, place_on / On, place_in / In or On(object), "
                    "slide / Open-Close, push_to / On(zone); text match: an atom argument shares a name token with "
                    "the step's part text", "version": PLAN_MATCH_VERSION}


def merge_plan_match(check, match):
    """M0's parse_check (agent_speedup_v2_truth, byte-identical: its atoms[].planned is always None) with the
    runner-side plan_match. plan_match is the authoritative per-atom field; its ``planned`` is copied into atoms[]
    where both list the same atoms in the same order, so a reader of the contract-level field sees the same value."""
    atoms = [dict(item) for item in check.get("atoms") or [] if isinstance(item, dict)]
    rows = [item for item in (match or {}).get("atoms") or [] if isinstance(item, dict)]
    aligned = len(atoms) == len(rows) and all(a.get("atom") == r.get("atom") for a, r in zip(atoms, rows))
    if aligned:
        for atom, row in zip(atoms, rows):
            atom["planned"] = row.get("planned")
    return {**check, "atoms": atoms, "plan_match": match,
            "planned_source": f"plan_match ({PLAN_MATCH_VERSION})" if aligned else None,
            "planned_note": "plan_match is authoritative; atoms[].planned is copied from it runner-side (the M0 "
                            "truth module leaves it None)" if aligned else
                            "atoms[] and plan_match list different atoms; atoms[].planned left as parse_check gave it"}


def resolve_expected_sha(task_key, episode_index, argument, *, injected_sim=False):
    expected = _validate_expected_sha(argument)
    if expected is not None:
        return expected, "argument"
    if injected_sim:     # test simulators render their own observations (as v1: no default for an injected sim)
        return None, None
    frozen = frozen_reset_sha256(task_key, episode_index)
    if frozen is not None:
        return frozen, "registry_frozen_reset"
    raise ValueError(f"no frozen reset sha256 for {task_key} episode {episode_index}; measure it and pass "
                     "--expected-initial-observation-sha256 (fail-closed)")


# ---------------------------------------------------------------------------------------------------------
# identity
def _file_sha(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def helper_chain_identity():
    return [_file_sha(path) for path in HELPER_CHAIN]


def lint_identity(libero_root):
    from gpt_field.agent_speedup_v2_tasks import list_v2_tasks
    vocab = thresholds.lint_vocabulary(libero_root, [spec.prompt for spec in list_v2_tasks()])
    return {"vocabulary_sha256": vocab["vocabulary_sha256"], "allowlist_sha256": vocab["allowlist_sha256"],
            "part_nouns_sha256": vocab["part_nouns_sha256"], "vocabulary": vocab["vocabulary"],
            "allowlist": vocab["allowlist"], "part_nouns": vocab["part_nouns"], "bddl_files": vocab["bddl_files"]}


@functools.lru_cache(maxsize=4)
def _lint_report(libero_root, require_all=True):
    """T0 lint of every control and model-interface module (cached per process: ~2 s)."""
    return thresholds.lint_control_modules(libero_root, require_all=require_all)


def lint_check_summary(libero_root, require_all=True):
    """Compact T0 identity (contract §C14.3 item 5: require_all): passed, missing modules, failing modules."""
    report = _lint_report(str(libero_root), require_all)
    failing = {name: [item.get("rule") for item in found]
               for group in ("modules", "interface_modules") for name, found in report[group].items() if found}
    return {"passed": report["passed"], "require_all": report["require_all"], "missing": list(report["missing"]),
            "failing": failing, "vocabulary_sha256": report["vocabulary_sha256"],
            "allowlist_sha256": report["allowlist_sha256"], "part_nouns_sha256": report["part_nouns_sha256"],
            "vocabulary_size": report["vocabulary_size"]}


def _v2_identity_hashes():
    """instructions / plan-schema hashes of the v2 client (None when a module cannot be imported)."""
    out = {"instructions_sha256": None, "plan_schema_sha256": None, "client_version": None, "prompt_version": None,
           "plan_schema_version": None, "errors": {}}
    try:
        from gpt_field import agent_speedup_v2_client as client
        out.update(instructions_sha256=client.instructions_sha256(), client_version=client.CLIENT_VERSION,
                   prompt_version=client.PROMPT_VERSION)
    except Exception as exc:  # pragma: no cover - a broken client module is reported, the smoke still runs
        out["errors"]["client"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from gpt_field import agent_speedup_v2_plan as plan
        out.update(plan_schema_sha256=plan.schemas_sha256(), plan_schema_version=plan.PLAN_SCHEMA_VERSION)
    except Exception as exc:  # pragma: no cover
        out["errors"]["plan"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def agent_metadata(libero_root):
    """Identity block shared by every registry key (AC-1: same constants, same instruction, same code)."""
    hashes = _v2_identity_hashes()
    return {"runner_version": RUNNER_VERSION, "robot": robot.robot_identity(),
            "rulings_assumed": dict(robot.RULINGS),
            "constants_sha256": thresholds.constants_sha256(), "constant_labels": thresholds.constants_register(),
            "thresholds_version": thresholds.THRESHOLDS_VERSION, "lint": lint_identity(libero_root),
            "instructions_sha256": hashes["instructions_sha256"], "plan_schema_sha256": hashes["plan_schema_sha256"],
            "client_version": hashes["client_version"], "prompt_version": hashes["prompt_version"],
            "plan_schema_version": hashes["plan_schema_version"], "identity_errors": hashes["errors"],
            "client_status": "AgentSpeedUpV2Client: plan / point / repair / review schemas (M5)",
            "model_role": "relation planner and part pointing; never numbers, directions or completion (AC-5)",
            "model_declares_completion": False, "contact_quantities_from_model": False,
            "policy_task_inputs": "none: the client's only task input is the official sentence read from the reset "
                                  "observation and checked against list_tasks()",
            "live_gate": LIVE_GATE}


# ---------------------------------------------------------------------------------------------------------
# offline client and the plumbing policy (M0; no model)
class OfflinePlumbingClient:
    """Stand-in client for the zero-call plumbing smoke. Like the v2 client (spec §4.2) it requires the episode's
    official sentence as ``task_prompt`` (no default) and checks every observation against it with the base
    client chain's prompt check (direct_history._allowed); it makes no request."""

    prompt_version = "agent-speedup-v2-offline-plumbing"
    reasoning_effort = None
    offline_mock = True

    def __init__(self, config=None, *, task_prompt, token_provider=None, transport=None, log_dir=None):
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise ValueError("task_prompt must be the episode's official sentence")
        self.config, self.task_prompt, self.log_dir = config, task_prompt, log_dir
        self.token_provider, self.transport = token_provider, transport
        self._used_access_tokens = set()
        self.checked_observations = 0

    def check_observation(self, observation):
        from gpt_field.direct_history import _allowed
        _allowed(observation.to_dict() if hasattr(observation, "to_dict") else observation, self.task_prompt)
        self.checked_observations += 1
        return True

    def budget_snapshot(self):
        return {"billing": "offline_mock", "api_calls": 0, "api_key_calls": 0, "subscription_inference_calls": 0,
                "mock_response_calls": 0}

    def history_stats(self):
        return {"protocol": "stateless_offline_plumbing", "accepted_requests": 0, "messages": 0,
                "retained_image_pairs": 0, "calls_by_kind": {}, "observations_checked": self.checked_observations}


PLUMBING_SCRIPT = (
    # (phase, steps, action) -- robot-only motions; no object interaction (gripper -1 open, +1 close)
    ("hold_open", 10, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)),
    ("empty_close", 20, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    ("open", 20, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)),
    ("lift", 8, (0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -1.0)),
    ("lower", 8, (0.0, 0.0, -0.2, 0.0, 0.0, 0.0, -1.0)),
    ("yaw_plus", 8, (0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0)),
    ("yaw_minus", 8, (0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -1.0)),
)


class PlumbingHoldPolicy:
    """Zero-call M0 smoke policy: the scripted robot-only phases, then hold with open fingers. It takes no task
    key; its outcomes exercise the runner's outcome, ledger and stuck-alert plumbing."""

    controller = CONTROLLER

    def __init__(self, client):
        self.client = client
        self.reset()

    def reset(self):
        self.index, self.ledger, self.outcomes, self.gap_trace, self._pending = 0, [], [], [], None
        self._phase_start_gap = None

    def _phase(self, index):
        start = 0
        for name, steps, action in PLUMBING_SCRIPT:
            if index < start + steps:
                return name, index - start, steps, action
            start += steps
        return "hold", index - start, None, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)

    def act(self, observation):
        if self._pending is not None:
            raise RuntimeError("act() called before observe_transition() of the previous action")
        self.client.check_observation(observation)
        name, k, steps, action = self._phase(self.index)
        outcomes = []
        if k == 0 and self.index > 0:
            previous, _, previous_steps, _ = self._phase(self.index - 1)
            gap = (observation.state[6] - observation.state[7]) * 1000.0
            record = {"n": len(self.outcomes) + 1, "relation": "plumbing", "skill": previous, "status": "succeeded",
                      "reason": "scripted_phase_complete", "step": self.index,
                      "evidence": {"finger_gap_mm": round(gap, 3), "steps": previous_steps}, "thresholds": []}
            self.outcomes.append(record)
            self.ledger.append({"skill": previous, "status": "succeeded"})
            outcomes.append(record)
        self._pending = self.index
        diagnostics = {"step": self.index, "controller": CONTROLLER, "macro": name, "primitive": "scripted_hold",
                       "execution_kind": f"plumbing:{name}", "macro_step": k, "refreshed": False, "routing": None,
                       "api_calls_in_act": [], "api_latency_s": None, "api_request_key": None,
                       "api_request_log": None, "pointing": [], "outcomes": outcomes,
                       "history_stats": self.client.history_stats()}
        return Decision(tuple(float(v) for v in action), diagnostics)

    def observe_transition(self, before, decision, after):
        if self._pending is None:
            raise RuntimeError("observe_transition() without a pending action")
        self._pending = None
        gap = (after.state[6] - after.state[7]) * 1000.0
        self.gap_trace.append(round(gap, 4))
        self.index += 1
        return {"finger_gap_mm": round(gap, 4),
                "grip_delta_mm": [round((a - b) * 1000.0, 3) for a, b in zip(after.state[:3], before.state[:3])]}

    def agent_stats(self):
        return {"policy": "PlumbingHoldPolicy", "steps": self.index, "outcomes": len(self.outcomes),
                "gap_trace_mm": list(self.gap_trace), "script": [[n, s] for n, s, _ in PLUMBING_SCRIPT]}

    @property
    def calibration(self):
        return None


def _default_client_factory(smoke, *, live=False):
    """plumbing smoke -> OfflinePlumbingClient; live (only on an explicit live run, ``live=True``) -> the v2 client
    with its default ordered SSE transport. The relation smoke's ORACLE client is built after the simulator starts
    (``_relation_smoke_client_factory``)."""
    if smoke:
        return OfflinePlumbingClient
    if live is not True:
        raise ValueError("the live agent_speedup_v2 client is handed out only to an explicit live run (--live; "
                         "execute(..., smoke=False) passes live=True)")
    from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client
    return AgentSpeedUpV2Client


def _oracle_module():
    """scripts/agent_speedup_v2_oracle.py, loaded only by the relation smoke (never at import, never live)."""
    name = "agent_speedup_v2_oracle"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _default_oracle_factory(sim, task_key):
    return _oracle_module().OracleAnswers(sim, task_key)


def _relation_smoke_client_factory(sim, task_key, oracle_factory=None):
    """The v2 client behind ScriptedV2Transport(ORACLE answers) for --smoke-script relation (contract §C14.3 item
    1); built after sim.start() and reset. It keeps the runner's client construction signature."""
    from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, ScriptedV2Transport
    answers = (oracle_factory or _default_oracle_factory)(sim, task_key)

    def factory(config=None, *, token_provider, task_prompt, transport=None, log_dir=None):
        if transport is not None:
            raise ValueError("the relation smoke's transport is the ORACLE's ScriptedV2Transport")
        return AgentSpeedUpV2Client(config, task_prompt=task_prompt, token_provider=token_provider,
                                    transport=ScriptedV2Transport(answers), log_dir=log_dir)
    factory.answers = answers
    factory.__qualname__ = "AgentSpeedUpV2Client[ScriptedV2Transport(OracleAnswers)]"
    return factory


def _plumbing_policy_factory(client):
    return PlumbingHoldPolicy(client)


def _v2_policy_factory(client):
    """The v2 policy, exactly as a live run builds it: AgentSpeedUpV2Policy(client), no overrides (§C13.1)."""
    from gpt_field.agent_speedup_v2 import AgentSpeedUpV2Policy
    return AgentSpeedUpV2Policy(client)


def _default_policy_factory(client):
    """M0's default (the plumbing smoke's policy); live and relation runs use ``_v2_policy_factory``."""
    return _plumbing_policy_factory(client)


def _default_recorder_factory():
    from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
    return AgentSpeedUpV2VideoRecorder


def _default_sim_factory(monitor_options):
    def factory(config):
        return V2SimProcess(config, monitor_options=monitor_options)
    return factory


# ---------------------------------------------------------------------------------------------------------
# preflight
LIVE_COMMAND_OPTIONS = ("optional flags: --store-dir DIR and --profile NAME (the ChatGPT subscription credential "
                        "store; default: chatgpt_auth's store, as v1); --stop-after-steps N (an operator stop at the "
                        "first complete-feedback boundary after N actions, status user_stop)")


def live_command(task_key, episode_index, *, stop_after_steps=None):
    """The bare live command the user runs, copy-paste safe (printed by --preflight; the output directory must be
    fresh). The optional flags are listed separately (LIVE_COMMAND_OPTIONS), never inlined as placeholders."""
    command = (f"cd {ROOT} && env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 .venv/bin/python {RUNNER_PATH} "
               f"--live --task {task_key} --episode-index {episode_index} "
               f"--output runs/agent_speedup_v2_live_$(date -u +%Y%m%dT%H%M%SZ)")
    if stop_after_steps is not None:
        command += f" --stop-after-steps {_validate_stop_after_steps(stop_after_steps)}"
    return command


def live_build_check(task_prompt):
    """Zero-call build check before a scored live run: the real plan / point_fine / point_coarse / repair payloads
    are built for a synthetic observation (``preflight_payloads``, which also parses the client's worked example
    against the strict plan schema and validates it), and every response schema is built and hashed. It raises on a
    lint-clean but broken client or plan build (e.g. a PlanParseError from the worked example); the result (payload
    sha256 per kind, schema and instruction hashes) is recorded in the live manifest. No credentials or network."""
    from gpt_field import agent_speedup_v2_client as client_module, agent_speedup_v2_plan as plan_module
    payloads = client_module.preflight_payloads(task_prompt)
    schemas = plan_module.schemas()
    missing = [kind for kind in ("plan", "point_fine", "point_coarse", "repair") if kind not in payloads]
    if missing:
        raise ValueError(f"preflight_payloads built no {missing} payload")
    return {"passed": True, "payload_sha256": {name: digest(payload) for name, payload in sorted(payloads.items())},
            "schemas": sorted(schemas), "plan_schema_sha256": plan_module.schemas_sha256(),
            "plan_schema_version": plan_module.PLAN_SCHEMA_VERSION,
            "instructions_sha256": client_module.instructions_sha256(),
            "prompt_version": client_module.PROMPT_VERSION, "client_version": client_module.CLIENT_VERSION,
            "check": "preflight_payloads(task prompt) built and the worked example parsed and validated; schemas() "
                     "built; zero model calls"}


def _live_build_failure(exc):
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def preflight(config, sim_config, output=None, *, task_key=PRIMARY_TASK, episode_index=0, libero_root=None):
    """Identity, prompt check and the real v2 payloads for a synthetic 512 observation; no credentials, network,
    files or simulator."""
    episode_index = _validate_episode_index(episode_index)
    task_spec = resolve_v2_task(task_key)
    validate_configuration(config, sim_config, task_spec)
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh agent_speedup_v2 experiment requires a new output directory")
    from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, preflight_payloads
    size = sim_config.image_size
    legacy = image_to_data_url(np.zeros((size, size, 3), dtype=np.uint8), size=size, rotate=False)
    observation = PolicyObservation(legacy, legacy, PREFLIGHT_STATE, task_spec.prompt)
    client = AgentSpeedUpV2Client(config, task_prompt=task_spec.prompt,
                                  token_provider=_v1._preflight_token_provider,
                                  transport=_v1._no_network_transport)
    client.check_observation(observation)
    payloads = preflight_payloads(task_spec.prompt)
    from gpt_field.direct_history import TASK as V1_BOWL_TASK
    root = libero_root or sim_config.libero_root
    metadata = agent_metadata(root)
    lint = thresholds.lint_control_modules(root)
    return {"status": "preflight_only", "method": METHOD, "task": task_spec.prompt, "task_key": task_spec.key,
            "suite": task_spec.suite, "task_id": task_spec.task_id, "episode_index": episode_index,
            "task_spec": task_spec.to_dict(), "simulator": asdict(sim_config),
            "simulator_monitor": build_v2_sim_config(task_spec.key)[1], "api": asdict(config),
            "client_task_prompt": client.task_prompt,
            "client_prompt_check": "accepted by AgentSpeedUpV2Client.check_observation (direct_history._allowed)",
            "client": _client_identity(client, config),
            "registry_prompt_is_v1_bowl_default": (client.task_prompt == V1_BOWL_TASK
                                                   and task_spec.key != "bowl_plate"),
            "registry_prompt_check_note": "by construction (registry prompt vs direct_history.TASK); the runtime "
                                          "client prompt is the reset observation's sentence, tested for every key",
            "agent_speedup_v2": {k: v for k, v in metadata.items() if k not in ("constant_labels",)},
            "lint_check": lint, "lint_check_require_all": lint_check_summary(root, True),
            "helper_chain": helper_chain_identity(),
            "mount_decision_on_synthetic_state": robot.mount_decision(PREFLIGHT_STATE).to_dict(),
            "synthetic_observation": True, "synthetic_state": list(PREFLIGHT_STATE),
            "frozen_reset_sha256": frozen_reset_sha256(task_spec.key, episode_index),
            "reset_hash_policy": "fail-closed: registry frozen hash or --expected-initial-observation-sha256",
            "payloads": {name: _payload_preview(payload) for name, payload in payloads.items()},
            "payload_sha256": {name: digest(payload) for name, payload in payloads.items()},
            "payload_note": "real v2 payloads (uniform grey 512 images, the worked example in the repair status "
                            "block); images are shown as placeholders; nothing is sent",
            "render_environment": render_environment(), "api_calls": 0, "credentials_checked": False,
            "simulator_started": False, "files_written": False,
            "sim_python_exists": Path(sim_config.python_executable or "").is_file(),
            "libero_checkout_exists": (Path(sim_config.libero_root or "")
                                       / "libero/libero/benchmark/__init__.py").is_file(),
            "live_command": live_command(task_spec.key, episode_index),
            "live_command_first_run": live_command(task_spec.key, episode_index,
                                                   stop_after_steps=LIVE_FIRST_RUN_STOP_AFTER_STEPS),
            "live_command_options": LIVE_COMMAND_OPTIONS, "live_gate": LIVE_GATE,
            "stopping_rule": STOPPING_RULE, "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None}


# ---------------------------------------------------------------------------------------------------------
# execution
def _read_sources(attempts=3, extra_scripts=()):
    """Package sources plus the exact hashed bytes, re-read until stable; the v2 runner, v1's runner and v1's
    helper scripts (plus the ORACLE in the relation smoke); the robot facts data file."""
    directory = Path(_experiment.__file__).parent
    for _ in range(attempts):
        software = source_manifest()
        try:
            sources = {name: (directory / name).read_bytes() for name in software["source_sha256"]}
        except FileNotFoundError:
            continue
        if all(_sha256(data) == software["source_sha256"][name] for name, data in sources.items()):
            break
    else:
        raise RuntimeError("Package sources kept changing while freezing provenance")
    scripts = {RUNNER_PATH: Path(__file__).read_bytes(), V1_RUNNER_PATH: (ROOT / V1_RUNNER_PATH).read_bytes(),
               **{path: (ROOT / path).read_bytes() for path in V1_HELPER_SCRIPTS},
               **{path: (ROOT / path).read_bytes() for path in extra_scripts}}
    data = {path: (ROOT / path).read_bytes() for path in DATA_FILES}
    return software, sources, scripts, data


def _attribute(owner, name):
    value = getattr(owner, name, None)
    return value() if callable(value) else value


def _console_event(row, steps):
    """A compact console line for a v2 trace event (scalar fields only)."""
    line = {"event": "v2", "name": row.get("event"), "step": steps}
    for key, value in row.items():
        if key in ("event", "policy_step") or len(line) >= CONSOLE_FIELDS + 3:
            continue
        if isinstance(value, str):
            line[key] = value[:CONSOLE_TEXT_CHARS]
        elif value is None or isinstance(value, (bool, int, float)):
            line[key] = value
    if row.get("event") == "plan_validation" and isinstance(row.get("issues"), list):
        line["issues"] = [str((item or {}).get("code")) for item in row["issues"]][:CONSOLE_FIELDS]
    return line


def execute(output, *, config, sim_config, task_key, episode_index, monitor_options=None, token_provider=None,
            client_factory=None, policy_factory=None, sim_factory=None, recorder_factory=None,
            expected_initial_observation_sha256=None, video=True, smoke=False, smoke_steps=DEFAULT_SMOKE_STEPS,
            smoke_script="plumbing", libero_root=None, oracle_factory=None, stop_after_steps=None):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError("Fresh agent_speedup_v2 experiment requires a new output directory; no resume")
    episode_index = _validate_episode_index(episode_index)
    task_spec = resolve_v2_task(task_key)
    if smoke_script not in SMOKE_SCRIPTS:
        raise ValueError(f"smoke_script must be one of {SMOKE_SCRIPTS}")
    if type(video) is not bool or type(smoke) is not bool:
        raise ValueError("video and smoke must be booleans")
    smoke_steps = _validate_smoke_steps(smoke_steps) if smoke else None
    stop_after_steps = _validate_stop_after_steps(stop_after_steps)
    if smoke and stop_after_steps is not None:
        raise ValueError("--stop-after-steps is an operator stop of a live run; a smoke uses --smoke-steps")
    relation_smoke = smoke and smoke_script == "relation"
    if oracle_factory is not None and not relation_smoke:
        raise ValueError("the ORACLE is used only by --sim-smoke --smoke-script relation (refused in live mode)")
    validate_configuration(config, sim_config, task_spec)
    expected, expected_source = resolve_expected_sha(task_spec.key, episode_index,
                                                     expected_initial_observation_sha256,
                                                     injected_sim=sim_factory is not None)
    if relation_smoke and client_factory is None:
        from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client
        client_class, manifest_client_class = None, AgentSpeedUpV2Client     # built after the simulator starts
    else:
        client_class = client_factory or _default_client_factory(smoke, live=not smoke)
        manifest_client_class = client_class
    if smoke:
        if token_provider not in (None, _v1.offline_token_provider):
            raise ValueError("Scripted sim smoke uses only its offline OAuth fixture, never credentials")
        token_provider = _v1.offline_token_provider
    elif not callable(token_provider):
        raise MissingCredentials("Live agent_speedup_v2 execution requires a ChatGPT subscription token provider")
    monitor_options = dict(monitor_options if monitor_options is not None else build_v2_sim_config(task_spec.key)[1])
    if relation_smoke:
        monitor_options["oracle"] = True
    elif monitor_options.get("oracle", False) is not False:
        raise ValueError("the ORACLE op is refused outside --sim-smoke --smoke-script relation (live mode refuses it)")
    worker_session = "own_session" if sim_factory is None else "injected_sim_factory"
    sim_factory = sim_factory or _default_sim_factory(monitor_options)
    recorder_factory = recorder_factory or (_default_recorder_factory() if video else NullRecorder)
    policy_factory = policy_factory or (_plumbing_policy_factory if smoke_script == "plumbing" and smoke
                                        else _v2_policy_factory)
    libero_root = libero_root or sim_config.libero_root
    software, sources, scripts, data_files = _read_sources(extra_scripts=(ORACLE_PATH,) if relation_smoke else ())
    mode = SMOKE_MODE if smoke else LIVE_MODE
    billing = "offline_mock" if smoke else "chatgpt_subscription"
    metadata = agent_metadata(libero_root)
    lint_check = lint_check_summary(libero_root, True)
    build_check = None
    if not smoke:            # a scored live run: the zero-call client / plan build check, before any file is written
        try:
            build_check = live_build_check(task_spec.prompt)
        except Exception as exc:
            raise ValueError(f"live build check fails ({_live_build_failure(exc)}); a live run needs a working "
                             "client and plan build (see --preflight)") from exc
    stopping = ([f"{smoke_steps} scripted smoke actions", *STOPPING_RULE] if smoke else
                [*STOPPING_RULE, *([f"operator step cap: stop after {stop_after_steps} actions (user_stop)"]
                                   if stop_after_steps is not None else [])])
    smoke_policy = ("PlumbingHoldPolicy (robot-only scripted phases)" if smoke and not relation_smoke else
                    "AgentSpeedUpV2Policy(client) with the ORACLE stub behind ScriptedV2Transport" if relation_smoke
                    else None)
    identity = {
        "kind": "agent_speedup_v2_sim_smoke" if smoke else "agent_speedup_v2_full_task", "mode": mode,
        "method": METHOD, "task": task_spec.prompt, "task_id": task_spec.task_id, "suite": task_spec.suite,
        "episode_index": episode_index, "task_key": task_spec.key, "task_spec": task_spec.to_dict(),
        "simulator": asdict(sim_config), "simulator_monitor": dict(monitor_options), "api": asdict(config),
        "billing": billing, "software": software,
        "runner": {"path": RUNNER_PATH, "sha256": _sha256(scripts[RUNNER_PATH])},
        "runner_helpers": [{"path": path, "sha256": _sha256(scripts[path])} for path in (
            V1_RUNNER_PATH, *V1_HELPER_SCRIPTS, *((ORACLE_PATH,) if relation_smoke else ()))],
        "helper_chain": helper_chain_identity(),
        "data_files": [{"path": path, "sha256": _sha256(blob)} for path, blob in data_files.items()],
        "client": {"class_factory": getattr(client_class, "__qualname__", None) or (
                       "AgentSpeedUpV2Client[ScriptedV2Transport(OracleAnswers)]" if relation_smoke
                       else str(client_class)),
                   "prompt_version": getattr(manifest_client_class, "prompt_version", None),
                   "reasoning_effort": getattr(manifest_client_class, "reasoning_effort", None),
                   "transport": ("ScriptedV2Transport(OracleAnswers)" if relation_smoke else
                                 "none (offline plumbing)" if smoke else "default (ordered SSE subscription)"),
                   "task_prompt_source": "reset observation, checked against list_tasks()",
                   "constructed_after_reset": True, "identity_file": "client.json"},
        "policy": {"controller": CONTROLLER, "controllers": [CONTROLLER], "task_key_reaches_policy": False,
                   "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
                   "object_state_feedback_to_policy": False, "smoke_policy": smoke_policy,
                   "factory": getattr(policy_factory, "__qualname__", str(policy_factory)),
                   "overrides_passed": False, "identity_file": "policy.json"},
        "oracle": {"enabled": relation_smoke, "module": ORACLE_PATH if relation_smoke else None,
                   "refused_in_live_mode": True},
        "agent_speedup_v2": metadata, "lint_check": lint_check, "live_build_check": build_check,
        "zero_shot": True, "collision_policy": "record_only",
        "collision_monitor": CONTACT_MODE, "collision_v1_chained": sim_config.collision_check_mode != "none",
        "success_qualification": "raw_libero_goal", "physical_truth": "diagnosis_only (AC-2; affects_completion "
                                                                       "false)",
        "v2_success_rule": "raw LIBERO success AND strict physical truth (release-and-settle counterfactual when only "
                           "hold checks fail); the same rule in live and smoke mode (lead decision L5)",
        "primary_timing_metric": "control_without_video_s", "operator_stop_file": OPERATOR_STOP_FILE,
        "operator_stop_after_steps": stop_after_steps,
        "camera_render_size": IMAGE_SIZE, "policy_image_size": IMAGE_SIZE,
        "expected_initial_observation_sha256": expected, "expected_initial_observation_sha256_source": expected_source,
        "video": video, "render_environment": render_environment(), "simulator_worker_session": worker_session,
        "smoke_steps": smoke_steps, "smoke_script": smoke_script if smoke else None,
        "stopping_rule": stopping, "stop_on_model_complete": False, "model_completion_is_libero_success": False,
        "max_policy_steps": smoke_steps, "max_api_calls": None, "max_wall_time_s": None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "evaluation_scored": not smoke, "prior_budgets_inherited": False, "live_gate": LIVE_GATE}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
                                          "created_utc": datetime.now(timezone.utc).isoformat(),
                                          "simulation_pauses_for_api": True})
    frozen = output / "frozen_source"
    frozen.mkdir()
    for name, blob in sources.items():
        (frozen / name).write_bytes(blob)
    for path, blob in {**scripts, **data_files}.items():
        (frozen / Path(path).name).write_bytes(blob)
    trace_path = output / "trace.jsonl"
    stop_command = f"touch {shlex.quote(str(output / OPERATOR_STOP_FILE))}"
    print(f"operator stop (ends the run at the next complete-feedback boundary): {stop_command}",
          file=sys.stderr, flush=True)

    status, stage, error = "running", "starting", None
    caught, interrupted_stage, stop_source = None, None, None
    steps = feedback_count = close_actions = open_actions = neutral_actions = 0
    pending_action, first_close_event, success_seen, path_length = None, None, False, 0.0
    macro = primitive = relation = None
    controller = None
    outcomes = []
    client = policy = recorder = sim = None
    oracle_answers = None
    cleanup_errors, failed_reads = [], {}
    started = time.monotonic()
    control_started = control_stopped = None
    collision, collision_v1, mount = None, None, None
    seen_forbidden = set()
    timings = {key: 0.0 for key in ("decision_s", "model_pipeline_s", "simulator_step_s", "observe_transition_s",
                                    "video_render_s", "run_state_s")}
    model_latency_by_kind, model_calls_by_kind, model_failed_by_kind = {}, Counter(), Counter()
    v2_event_counts, runner_event_counts = Counter(), Counter()
    model_call_count = 0
    stuck_alert, stuck_alerts, stuck_armed, console_pending = None, [], True, None
    failures_since_alert, alerts_since_success = 0, 0
    recovery_stop = None                 # the policy's recovery_exhausted event row (ends the episode, ruling L3)
    collected, pre_video_written, signals = {}, False, None
    truth = {"terminal": None, "after_release_settle": None, "answer_key": None, "parse_check": None,
             "skipped": None}
    terminal_raw_libero_success = None

    def secrets():
        return list(getattr(client, "_used_access_tokens", ()) or ())

    def guarded(label, read, default=None):
        try:
            return _checked_json(read())
        except Exception as exc:
            if label not in failed_reads:
                failed_reads[label] = _safe_error(exc, label)
                _write_traceback(output, exc, label, secrets())
            return default

    def trace_error(exc, label):
        try:
            _write_traceback(output, exc, label, secrets())
        except Exception:
            pass

    def budget():
        return guarded("budget_snapshot", client.budget_snapshot, {}) if client is not None else {}

    def history():
        return guarded("history_stats", client.history_stats, {}) if client is not None else {}

    def state(new_stage):
        """run_state.json; the policy's agent_stats() is read once per step (at "observed") and at the start and
        finalization stages, not at "deciding" / "executing" (its cost is inside control time: run_state_s)."""
        nonlocal stage
        stage = new_stage
        began = time.monotonic()
        light = stage in ("deciding", "executing")
        write_json(output / "run_state.json", {
            "status": status, "stage": stage, "mode": mode, "method": METHOD, "task_key": task_spec.key,
            "task_id": task_spec.task_id, "episode_index": episode_index, "step": steps, "controller": controller,
            "macro": macro, "primitive": primitive, "relation": relation, "pending_action_index": pending_action,
            "relation_outcomes": len(outcomes), "model_calls": model_call_count, "budget": budget(),
            "history_stats": history(),
            "agent": (guarded("agent_stats", policy.agent_stats) if policy is not None and not light
                      else "read at the observed stage" if policy is not None else None),
            "collision": collision, "first_close_event": first_close_event, "error": error,
            "stuck_alert": stuck_alert, "operator_stop_command": stop_command})
        timings["run_state_s"] += time.monotonic() - began

    def wall_s():
        return round(time.monotonic() - started, 1)

    def console(row):
        print(json.dumps(row, separators=(",", ":"), default=str), flush=True)

    def raise_alert(reason, record, *, automatic_cap=False):
        nonlocal stuck_alert, stuck_armed, failures_since_alert, alerts_since_success
        stuck_armed = False
        if stuck_alert is not None and stuck_alert.get("active"):
            stuck_alert.update(active=False, superseded_by_outcome_n=record.get("n"), superseded_at_step=steps)
        failures_since_alert = 0
        stuck_alert = {"active": True, "reason": reason, "outcome_n": record.get("n"), "step": steps,
                       "wall_s": wall_s(), "api_calls": budget().get("api_calls"), "stop_command": stop_command,
                       "repeat": alerts_since_success, "automatic_cap": bool(automatic_cap)}
        alerts_since_success += 1
        stuck_alerts.append(stuck_alert)
        tail = ("the episode ends now (automatic cap: recovery_exhausted)" if automatic_cap else
                "the run continues (no automatic cap); to stop at the next complete-feedback boundary: "
                f"{stop_command}")
        print(f"STUCK ALERT: {reason} | step {steps} | wall {stuck_alert['wall_s']:.1f} s | api calls "
              f"{stuck_alert['api_calls']} | {tail}", file=sys.stderr, flush=True)
        state(stage)

    def note_outcome(record):
        nonlocal stuck_alert, stuck_armed, failures_since_alert, alerts_since_success
        console({"event": "outcome", **{key: record.get(key) for key in ("n", "skill", "status", "reason", "step")},
                 "pids": record.get("pids"), "wall_s": wall_s(), "api_calls": budget().get("api_calls")})
        immediate = _immediate_stuck_reason(record)
        if immediate is not None:
            raise_alert(immediate, record)
            return
        if not _stuck_counted(record):      # bookkeeping (a repoint's registration): neither progress nor failure
            return
        if record.get("status") == "succeeded":
            if not stuck_armed and stuck_alert is not None:
                stuck_alert.update(active=False, cleared_by_outcome_n=record.get("n"), cleared_at_step=steps)
            stuck_armed, failures_since_alert, alerts_since_success = True, 0, 0
            return
        if stuck_armed:
            reason = _stuck_reason(outcomes)
        else:            # an alert stands: re-alert every STUCK_REPEAT_FAILURES further failures (no cap)
            failures_since_alert += int(record.get("status") in ("failed", "rejected"))
            reason = _repeat_stuck_reason(failures_since_alert, stuck_alert)
        if reason is not None:
            raise_alert(reason, record)

    def note_model_calls(details):
        nonlocal model_call_count
        routing = details.get("routing") if isinstance(details.get("routing"), dict) else {}
        calls = [call for call in details.get("api_calls") or [] if isinstance(call, dict)]
        for index, call in enumerate(calls):
            kind = str(call.get("kind"))
            latency = float(call.get("latency_s") or 0.0)
            model_call_count += 1
            model_calls_by_kind[kind] += 1
            model_latency_by_kind[kind] = model_latency_by_kind.get(kind, 0.0) + latency
            if call.get("ok") is False:
                model_failed_by_kind[kind] += 1
            line = {"event": "model_call", "k": model_call_count, "kind": kind, "ok": call.get("ok"),
                    "latency_s": round(latency, 2), "step": steps, "wall_s": wall_s(),
                    "api_calls": budget().get("api_calls")}
            if index == len(calls) - 1 and routing:      # diagnostics carry the routing of the act's last Need
                params = routing.get("params") if isinstance(routing.get("params"), dict) else {}
                line.update(params={key: value for key, value in params.items() if value is not None},
                            situation=str(routing.get("situation") or "")[:CONSOLE_TEXT_CHARS] or None)
            console(line)

    def finalization_trace_events():
        known = {item.get("n") for item in outcomes if item.get("n") is not None}
        events = []
        for outcome in list(getattr(policy, "outcomes", None) or []):
            n = outcome.get("n") if isinstance(outcome, dict) else None
            if n is None or n in known:
                continue
            known.add(n)
            outcomes.append({"n": n, "step": outcome.get("step"), "skill": outcome.get("skill"),
                             "relation": outcome.get("relation"), "status": outcome.get("status"),
                             "reason": outcome.get("reason"), "pids": outcome.get("pids"),
                             "recorded_at_finalization": True})
            events.append({"event": "relation_outcome", "step": outcome.get("step"), "outcome": outcome,
                           "mode": mode, "recorded_at_finalization": True})
        if isinstance(caught, KeyboardInterrupt):
            events.append({"event": "operator_stop", "source": stop_source, "step": steps,
                           "feedback_count": feedback_count, "pending_action_index": pending_action,
                           "request_file": None, "stage": interrupted_stage})
        if events and trace_path.is_file():
            with trace_path.open("a") as log:
                for row in events:
                    _append(log, row)
        return len(events)

    def truth_read(label, read):
        value = guarded(label, read)
        if value is not None and not _truth_reply_shape_ok(label, value):
            failed_reads.setdefault(label, {"type": "TruthReplyShapeError", "stage": label,
                                            "message": f"the {label} reply does not match the op sent"})
            return None
        return value

    def relation_plan():
        return guarded("relation_plan", lambda: _attribute(policy, "relation_plan")) if policy is not None else None

    def finalize_truth():
        """Diagnosis-only (spec §6.5): terminal truth, the release-and-settle counterfactual, the answer key."""
        if sim is None or not hasattr(sim, "truth"):
            truth["skipped"] = "no simulator truth interface"
            return
        if status not in TRUTH_STATUSES or control_started is None:
            truth["skipped"] = f"status {status} (simulator state not trusted)"
            return
        if pending_action is not None or getattr(sim, "_in_flight", False):
            truth["skipped"] = "interrupted during a simulator request (reply in flight)"
            return
        state(TRUTH_STAGE)
        truth["terminal"] = truth_read("truth_terminal", sim.truth)
        truth["after_release_settle"] = (truth_read("truth_after_release_settle", sim.settle_truth)
                                         if truth["terminal"] is not None else None)
        truth["answer_key"] = truth_read("answer_key", sim.answer_key)
        if truth["terminal"] is not None:
            write_json(output / "physical_truth.json", {
                "diagnosis_only": True, "affects_completion": False, "terminal": truth["terminal"],
                "after_release_settle": truth["after_release_settle"],
                "summary": physical_summary(truth["terminal"], truth["after_release_settle"],
                                            raw_libero_success=terminal_raw_libero_success)})
        if truth["answer_key"] is not None:
            write_json(output / "answer_key.json", truth["answer_key"])
            plan = relation_plan()
            truth["parse_check"] = merge_plan_match(parse_check(truth["answer_key"], plan=plan),
                                                    plan_atom_check(truth["answer_key"], plan))
            write_json(output / "parse_check.json", truth["parse_check"])

    def collect():
        if not collected:
            calibration = guarded("calibration", lambda: _attribute(policy, "calibration")) if policy else None
            collected.update({
                "agent_stats": guarded("agent_stats", policy.agent_stats) if policy is not None else None,
                "ledger": guarded("ledger", lambda: list(_attribute(policy, "ledger") or [])) if policy else [],
                "calibration": calibration, "budget": budget() or {}, "history": history() or {},
                "relation_plan": relation_plan()})
        return collected

    def outcome_status(video_error=None):
        """v1's promotion of cleanup failures to infrastructure_error, minus the diagnosis-only truth stage."""
        infra = ([item for item in cleanup_errors if item.get("stage") != TRUTH_STAGE]
                 or [label for label in failed_reads if label not in TRUTH_READS] or video_error)
        if infra and status not in OUTCOME_STATUSES:
            return "infrastructure_error"
        return status

    def model_accounting(budget_data, history_data):
        offline = budget_data.get("billing") == "offline_mock" or bool(budget_data.get("offline_mock"))
        sent = budget_data.get("api_calls")
        return {"billing": budget_data.get("billing"), "offline_mock": offline,
                "requests_sent": sent, "real_model_calls": 0 if offline else sent,
                "calls_by_kind": history_data.get("calls_by_kind"),
                "accepted_by_kind": history_data.get("accepted_by_kind"),
                "runner_calls_by_kind": dict(model_calls_by_kind),
                "failed_calls_by_kind": dict(model_failed_by_kind),
                "latency_s_by_kind": {k: round(v, 6) for k, v in model_latency_by_kind.items()},
                "tokens": {"input": budget_data.get("input_tokens"),
                           "cached_input": budget_data.get("cached_input_tokens"),
                           "output": budget_data.get("output_tokens"),
                           "reasoning": budget_data.get("reasoning_tokens"),
                           "unknown_usage_calls": budget_data.get("unknown_usage_calls"),
                           "output_by_kind": history_data.get("output_tokens_by_kind"),
                           "reasoning_by_kind": history_data.get("reasoning_tokens_by_kind"),
                           "visible_by_kind": history_data.get("visible_tokens_by_kind"),
                           "visible_budgets": history_data.get("visible_token_budgets"),
                           "over_visible_budget_by_kind": history_data.get("over_visible_budget_by_kind")},
                "overload_retry_requests": history_data.get("overload_retry_requests"),
                "overload_retry_wait_s": history_data.get("overload_retry_wait_s")}

    def build_summary(finalization_stage, video_error=None):
        data = collect()
        ledger, budget_data, history_data = data["ledger"], data["budget"], data["history"]
        final = outcome_status(video_error)
        completed = not smoke and final == "success"
        physical = (physical_summary(truth["terminal"], truth["after_release_settle"],
                                     raw_libero_success=terminal_raw_libero_success)
                    if truth["terminal"] is not None else
                    {"verdict": "truth_unavailable", "skipped": truth["skipped"], "diagnosis_only": True,
                     "affects_completion": False})
        v2_success = bool(final in V2_SUCCESS_STATUSES and terminal_raw_libero_success is True
                          and physical.get("v2_success") is True)
        policy_compute = timings["decision_s"] - timings["model_pipeline_s"]
        accounting = model_accounting(budget_data, history_data)
        agent = data["agent_stats"] if isinstance(data["agent_stats"], dict) else {}
        present = set(v2_event_counts) | {"threshold_set"}       # the nominal names are v2 trace events (§C4.8)
        from gpt_field.agent_speedup_v2_types import NOMINAL_EVENTS
        return {
            "status": final, "mode": mode, "method": METHOD, "task": task_spec.prompt, "suite": task_spec.suite,
            "task_id": task_spec.task_id, "task_key": task_spec.key, "episode_index": episode_index, "seed": 7,
            "requested_episodes": 1, "steps": steps, "calls_by_kind": history_data.get("calls_by_kind"),
            "relation_outcomes": len(outcomes),
            "relation_outcome_counts": dict(Counter(f"{item.get('skill')}:{item.get('status')}" for item in outcomes)),
            "relation_outcome_sequence": list(outcomes),
            "outcomes_recorded_at_finalization": sum(bool(item.get("recorded_at_finalization")) for item in outcomes),
            "relation_plan": data["relation_plan"], "parse_check": truth["parse_check"],
            "plan_steps": agent.get("steps"), "step_budget": agent.get("step_budget"),
            "v2_event_counts": dict(v2_event_counts),
            "runner_pointing_events": runner_event_counts["pointing_result"],
            "nominal_events_missing": [name for name in NOMINAL_EVENTS if name not in present],
            "final_controller": controller, "final_macro": macro, "final_primitive": primitive,
            "final_relation": relation,
            "measured_feedback_count": feedback_count, "gripper_close_actions": close_actions,
            "gripper_open_actions": open_actions, "gripper_neutral_actions": neutral_actions,
            "path_length_m": path_length, "pending_action_index": pending_action,
            "first_close_event": first_close_event, "calibration": data["calibration"],
            "mount_decision": mount, "ledger_entries": len(ledger) if isinstance(ledger, list) else None,
            "agent_stats": "agent_stats.json", "ledger": "ledger.json",
            "diagnostic_complete": smoke and final in ("diagnostic_complete", "diagnostic_sim_terminated"),
            "smoke_steps": smoke_steps, "smoke_script": smoke_script if smoke else None,
            "operator_stop_after_steps": stop_after_steps,
            "stop_on_model_complete": False, "model_declared_complete": False, "model_completion": None,
            "benchmark_completed": completed, "reasoning_effort": getattr(client, "reasoning_effort", None),
            "prompt_version": getattr(client, "prompt_version", None), "client_task_prompt":
                getattr(client, "task_prompt", None),
            "timing_components": {**timings, "policy_compute_s": policy_compute,
                                  "model_pipeline_s_by_kind": {k: round(v, 6)
                                                               for k, v in model_latency_by_kind.items()}},
            "timing_components_note": "model_pipeline_s is included in decision_s; policy_compute_s = decision_s - "
                "model_pipeline_s; model_pipeline_s_by_kind splits model_pipeline_s by request kind "
                "(diagnostics api_calls); observe_transition_s, simulator_step_s and video_render_s are separate; "
                "run_state_s (run_state.json writes, inside control time) is informational. "
                "control_without_video_s = control_elapsed_s - video_render_s. Terminal truth, video and cleanup "
                "are excluded from control_elapsed_s.",
            "libero_success_observed": success_seen, "libero_success": terminal_raw_libero_success,
            "task_success": completed, "v2_success": v2_success,
            "v2_success_rule": identity["v2_success_rule"],
            "physical_truth": physical, "collision": collision, "collision_v1_record_only": collision_v1,
            "completion_time_eligible": completed, "failure_reason": None, "evaluation_completed": completed,
            "success_rate": 1.0 if completed else None, "scored_episodes": int(completed),
            "incomplete_episodes": int(not completed), "evaluation_scored": not smoke,
            "api_calls": 0 if smoke else budget_data.get("api_calls"),
            "mock_response_calls": (budget_data.get("mock_response_calls", budget_data.get("api_calls", 0))
                                    or 0) if smoke else 0,
            "model_calls": accounting,
            "billing": billing, "budget": budget_data, "history_stats": history_data,
            "api_key_calls": budget_data.get("api_key_calls", 0), "api_key_fallback": False,
            "max_policy_steps": smoke_steps, "max_api_calls": None, "max_wall_time_s": None,
            "elapsed_s": time.monotonic() - started,
            "startup_to_initial_observation_s": (control_started - started if control_started is not None else None),
            "control_elapsed_s": (control_stopped - control_started if control_started is not None else None),
            "control_without_video_s": (control_stopped - control_started - timings["video_render_s"]
                                        if control_started is not None else None),
            "video": "episode.mp4" if (output / "episode.mp4").is_file() else None, "video_enabled": video,
            "video_partial": "episode.partial.mp4" if (output / "episode.partial.mp4").is_file() else None,
            "trace": "trace.jsonl", "error": error,
            "error_traceback": TRACEBACK_FILE if (output / TRACEBACK_FILE).is_file() else None,
            "cleanup_errors": [*cleanup_errors, *failed_reads.values()], "video_error": video_error,
            "stuck_alerts": list(stuck_alerts), "operator_stop_command": stop_command,
            "oracle": {"enabled": relation_smoke, "report": "oracle_report.json"
                       if (output / "oracle_report.json").is_file() else None},
            "lint_check": lint_check, "live_gate": LIVE_GATE,
            "finalization": {"stage": finalization_stage,
                             "deferred_signals": list(signals.deferred) if signals is not None else []},
            "output": str(output),
            "interpretation": ("Scripted agent_speedup_v2 plumbing smoke with real simulator execution; no model "
                               "calls and no performance score." if smoke and not relation_smoke else
                               "agent_speedup_v2 relation smoke: the real v2 policy, client and relation layer on "
                               "the real simulator with the ORACLE stub answering every request (zero model calls, "
                               "unscored). summary.v2_success = raw LIBERO success AND strict physical truth (L5)."
                               if relation_smoke else
                               "One zero-shot agent_speedup_v2 episode. Only raw LIBERO success completes the "
                               "evaluation; v2_success additionally needs the diagnosis-only strict physical truth "
                               "(AC-2). Collisions are evaluator-only secondary metrics.")}

    def write_artifacts(summary):
        data = collect()
        write_json(output / "runner_summary.json", summary)
        write_json(output / "summary.json", summary)
        if collision is not None:
            write_json(output / "collision.json", {"generic": collision, "v1_record_only": collision_v1})
        write_json(output / "task_evaluation.json", {key: summary[key] for key in (
            "status", "task_success", "libero_success", "collision", "failure_reason", "evaluation_completed",
            "success_rate", "scored_episodes", "incomplete_episodes", "completion_time_eligible")})
        write_json(output / "agent_stats.json", data["agent_stats"])
        write_json(output / "ledger.json", data["ledger"])

    def write_oracle_report():
        if oracle_answers is not None and callable(getattr(oracle_answers, "report", None)):
            report = guarded("oracle_report", oracle_answers.report)
            if report is not None:
                write_json(output / "oracle_report.json", report)

    def abort_finalization():
        if not pre_video_written:
            write_artifacts(build_summary("aborted_before_video_finalization"))
        state("finalization_aborted")

    def record_collisions(append, step_index):
        nonlocal collision, collision_v1
        collision = _collision_record_v2(sim.status)
        collision_v1 = _collision_v1_record(sim.status, identity["collision_v1_chained"])
        new = [name for name in (collision.get("first_by_class") or {}) if name in FORBIDDEN
               and name not in seen_forbidden]
        for name in new:
            seen_forbidden.add(name)
            append({"event": "recorded_collision", "step": step_index, "class": name,
                    "first": collision["first_by_class"][name], "first_detection": len(seen_forbidden) == 1,
                    "collision_policy": "record_only", "libero_success": sim.status["libero_success"],
                    "affects_completion": False})

    def v2_event_rows(details, append):
        """Each v2 event becomes one trace row (its only copy: the action row omits diagnostics["v2_events"]); a
        payload ``step`` / ``mode`` is kept as ``event_step`` / ``event_mode`` beside the runner's step and mode."""
        written = 0
        for row in details.get("v2_events") or []:
            if not isinstance(row, dict) or not row.get("event"):
                continue
            name = str(row["event"])
            v2_event_counts[name] += 1
            record = {"event": name, "step": steps,
                      **{key: value for key, value in row.items() if key not in ("event", "step", "mode")},
                      "mode": mode}
            if "step" in row:
                record["event_step"] = row["step"]
            if "mode" in row:
                record["event_mode"] = row["mode"]
            append(record)
            written += 1
            if name in CONSOLE_EVENTS:
                console(_console_event(row, steps))
        return written

    with termination_as_interrupt():
        try:
            state("starting")
            with _simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            if not relation_smoke and (getattr(sim, "metadata", None) or {}).get("oracle_enabled") is True:
                raise ValueError("the simulator worker has the ORACLE op enabled; refused outside the relation smoke")
            tasks = sim.list_tasks()
            selected = [task for task in tasks if task.get("task_id") == task_spec.task_id]
            if len(selected) != 1 or selected[0].get("prompt") != task_spec.prompt:
                raise ValueError("Official task does not match the selected task profile")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(task_spec.task_id, episode_index)
            collision = _collision_record_v2(sim.status)
            if obs.prompt != task_spec.prompt or obs.prompt != selected[0].get("prompt"):
                raise ValueError("Reset observation has a different task")
            initial_observation = obs.to_dict()
            initial_sha256 = initial_observation_sha256(initial_observation)
            write_json(output / "initial_observation.json", initial_observation)
            write_json(output / "initial_observation_check.json", {
                "sha256": initial_sha256, "expected_sha256": expected, "expected_source": expected_source,
                "matched": initial_sha256 == expected if expected else None})
            if expected and initial_sha256 != expected:
                raise ValueError("Reset observation differs from the frozen reset; no inference performed")
            if client_class is None:            # the relation smoke: the ORACLE needs the started simulator
                client_class = _relation_smoke_client_factory(sim, task_spec.key, oracle_factory)
                oracle_answers = client_class.answers
            # The client's only task input: the official sentence of this reset observation (spec §4.2).
            client = client_class(config, token_provider=token_provider, task_prompt=obs.prompt,
                                  transport=None, log_dir=output / "requests")
            if not smoke and getattr(client, "offline_mock", False) is True:
                print("warning: the live client reports offline_mock (an injected scripted transport); billing is "
                      "recorded from the client's own budget snapshot", file=sys.stderr, flush=True)
            write_json(output / "client.json", _client_identity(client, config))
            policy = policy_factory(client)
            write_json(output / "policy.json", _policy_identity(policy))
            mount = robot.mount_decision(obs.state).to_dict()
            policy.reset()
            label = ("SCRIPTED SIM SMOKE" if smoke and not relation_smoke else
                     "ORACLE RELATION SMOKE" if relation_smoke else "CHATGPT SUBSCRIPTION")
            recorder = recorder_factory(output / "episode.mp4", episode_id=(
                f"{label} / {task_spec.suite}_t{task_spec.task_id:02d}_i{episode_index:02d} / AGENT SPEEDUP V2 / "
                + ("NO MODEL CALLS" if smoke else "COLLISIONS RECORDED")), offline_smoke=smoke)
            recorder.observe(obs, step=0)
            control_started = time.monotonic()
            with trace_path.open("w") as log:
                def append(row):
                    _append(log, row)
                append({"event": "reset", "task_id": task_spec.task_id, "episode_index": episode_index,
                        "observation": obs.to_dict(), "sim_status": dict(sim.status), "mode": mode})
                append({"event": "threshold_set", "constants_sha256": thresholds.constants_sha256(),
                        "constant_labels": thresholds.constants_register(),
                        "robot_thresholds": [t.to_dict() for t in robot.robot_thresholds()],
                        "mount_decision": mount, "rulings": dict(robot.RULINGS)})
                record_collisions(append, None)
                while True:
                    collision = _collision_record_v2(sim.status)
                    success_seen = success_seen or sim.status["libero_success"]
                    if sim.status.get("success"):
                        success_seen = True
                        status = "diagnostic_sim_terminated" if smoke else "success"
                        break
                    if (output / OPERATOR_STOP_FILE).is_file():
                        status = "user_stop"
                        error = {"type": "OperatorStopRequested", "stage": stage, "source": "file",
                                 "message": "Operator stop requested; stopped at a complete-feedback boundary"}
                        append({"event": "operator_stop", "source": "file", "step": steps,
                                "feedback_count": feedback_count, "pending_action_index": pending_action,
                                "request_file": OPERATOR_STOP_FILE, "stage": stage})
                        break
                    if stop_after_steps is not None and steps >= stop_after_steps:
                        status = "user_stop"
                        error = {"type": "OperatorStopRequested", "stage": stage, "source": "step_cap",
                                 "message": f"Operator step cap ({stop_after_steps} actions) reached; stopped at a "
                                            "complete-feedback boundary"}
                        append({"event": "operator_stop", "source": "step_cap", "step": steps,
                                "feedback_count": feedback_count, "pending_action_index": pending_action,
                                "request_file": None, "stage": stage, "stop_after_steps": stop_after_steps})
                        break
                    if recovery_stop is not None:            # ruling L3: the policy's recovery ladder ended
                        status = RECOVERY_EXHAUSTED_STATUS
                        error = {"type": "RecoveryExhausted", "stage": stage, "source": "policy",
                                 "message": "the policy's repair ladder skipped a step and nothing can run; the "
                                            "episode ends as a failure (automatic cap)"}
                        append({"event": "recovery_stop", "step": steps, "feedback_count": feedback_count,
                                "pending_action_index": pending_action, "policy_event": recovery_stop,
                                "stage": stage})
                        break
                    if smoke and steps >= smoke_steps:
                        status = "diagnostic_complete"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated without task success or an action horizon")
                    if obs.prompt != task_spec.prompt:
                        raise ValueError("Policy task changed during rollout")
                    state("deciding")
                    began = time.monotonic()
                    decision = policy.act(obs)
                    latency = time.monotonic() - began
                    details = decision.diagnostics
                    timings["decision_s"] += latency
                    timings["model_pipeline_s"] += float(details.get("api_latency_s") or 0.0)
                    controller = details["controller"]
                    if controller != CONTROLLER:
                        raise ValueError(f"agent_speedup_v2 decisions must use the {CONTROLLER} controller")
                    macro, primitive = details.get("macro"), details.get("primitive")
                    relation = details.get("relation")
                    action = [float(value) for value in decision.action]
                    note_model_calls(details)
                    v2_rows = v2_event_rows(details, append)
                    exhausted = next((row for row in details.get("v2_events") or [] if isinstance(row, dict)
                                      and row.get("event") == RECOVERY_EXHAUSTED_EVENT), None)
                    if exhausted is not None and recovery_stop is None:
                        recovery_stop = exhausted
                        raise_alert("recovery_exhausted: the policy's repair ladder skipped a step and nothing can "
                                    "run any more", {"n": None}, automatic_cap=True)
                    for outcome in details.get("outcomes") or []:
                        record = {"n": outcome.get("n"), "step": steps, "skill": outcome.get("skill"),
                                  "relation": outcome.get("relation"), "status": outcome.get("status"),
                                  "reason": outcome.get("reason"), "pids": outcome.get("pids"),
                                  "plan_step": outcome.get("plan_step")}
                        outcomes.append(record)
                        append({"event": "relation_outcome", "step": steps, "outcome": outcome, "mode": mode})
                        note_outcome(record)
                    # The act's pointing results (diagnostics["pointing"]) under their own name: the v2 "pointing"
                    # event (a relation-layer trace event) is a different row schema.
                    for index, result in enumerate(details.get("pointing") or []):
                        runner_event_counts["pointing_result"] += 1
                        append({"event": "pointing_result", "step": steps, "index": index, "result": result,
                                "mode": mode})
                    append({"event": "action", "step": steps, "state": list(obs.state), "action": action,
                            "diagnostics": {key: value for key, value in details.items() if key != "v2_events"},
                            "v2_event_rows": v2_rows, "decision_latency_s": latency, "mode": mode})
                    video_began = time.monotonic()
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    timings["video_render_s"] += time.monotonic() - video_began
                    pending_action = steps
                    state("executing")
                    before = obs
                    sim_began = time.monotonic()
                    obs, success = sim.step(decision.action)
                    timings["simulator_step_s"] += time.monotonic() - sim_began
                    executed_step, steps = steps, steps + 1
                    pending_action = None
                    if success != sim.status["success"]:
                        raise ValueError("Step result must match the configured simulator success criterion")
                    success_seen = success_seen or sim.status["libero_success"]
                    close_actions += int(action[6] > 0)
                    open_actions += int(action[6] < 0)
                    neutral_actions += int(action[6] == 0)
                    path_length += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    append({"event": "executed", "step": executed_step, "controller": controller, "macro": macro,
                            "primitive": primitive, "next_state": list(obs.state), "observation": obs.to_dict(),
                            "action": action, "success": success, "sim_status": dict(sim.status), "mode": mode})
                    record_collisions(append, executed_step)
                    # Calibration closes are not the first grasp close (v1 excludes macro "calibrate"); the v2
                    # policy marks its Calibrate+ closes with diagnostics["calibration_probe"] (M6).
                    if first_close_event is None and action[6] > 0 and not details.get("calibration_probe"):
                        first_close_event = {"action_index": executed_step, "action_ordinal": steps,
                                             "controller": controller, "macro": macro, "primitive": primitive,
                                             "gripper_joint_delta": [obs.state[i] - before.state[i] for i in (6, 7)],
                                             "full_closure_verified": False, "grasp_verified": False}
                    feedback_began = time.monotonic()
                    feedback = policy.observe_transition(before, decision, obs)
                    timings["observe_transition_s"] += time.monotonic() - feedback_began
                    feedback_count += 1
                    append({"event": "feedback", "step": executed_step, "feedback": feedback})
                    video_began = time.monotonic()
                    recorder.executed(obs, step=executed_step, success=success, sim_status=dict(sim.status),
                                      feedback=feedback, history_stats=client.history_stats())
                    timings["video_render_s"] += time.monotonic() - video_began
                    if success:
                        status = "diagnostic_sim_terminated" if smoke else "success"
                    state("observed")
                    line = {"event": "executed", "mode": mode, "steps": steps, "controller": controller,
                            "macro": macro, "primitive": primitive, "close_actions": close_actions,
                            "success_observed": success, "api_attempts": budget().get("api_calls"),
                            "forbidden_contact_points": (collision.get("counts") or {}).get("forbidden_contact_points")}
                    if success or steps % EXECUTED_CONSOLE_EVERY == 0:
                        print(json.dumps(line), flush=True)
                        console_pending = None
                    else:
                        console_pending = line
                    if success:
                        break
        except KeyboardInterrupt as exc:
            stop_source = "sigterm" if str(exc) == f"Received signal {int(signal.SIGTERM)}" else "sigint"
            status, caught, interrupted_stage = "user_stop", exc, stage
            error = {"type": "KeyboardInterrupt", "stage": stage, "source": stop_source,
                     "message": "User interrupted the agent_speedup_v2 episode"}
        except (APIError, MissingCredentials, BudgetExceeded, SchemaError) as exc:
            status, error, caught = "subscription_error", _safe_error(exc, stage), exc
        except Exception as exc:
            status, error, caught = "infrastructure_error", _safe_error(exc, stage), exc
        finally:
            if control_started is not None and control_stopped is None:
                control_stopped = time.monotonic()
            signals = _FinalizationSignals(presses=int(isinstance(caught, KeyboardInterrupt)),
                                           on_abort=abort_finalization).install()
            summary = None
            try:
                if caught is not None:
                    trace_error(caught, interrupted_stage or stage)
                try:
                    terminal_raw_libero_success = sim.status.get("libero_success") if sim is not None else None
                except Exception:
                    terminal_raw_libero_success = None
                guarded("finalization_trace_events", finalization_trace_events, 0)
                # Diagnosis-only truth: after the terminal status is fixed, before the simulator closes.
                try:
                    finalize_truth()
                except Exception as exc:  # recorded, never re-raised, never changes the outcome
                    cleanup_errors.append(_error_record(exc, TRUTH_STAGE, secrets()))
                    trace_error(exc, TRUTH_STAGE)
                write_oracle_report()
                if sim is not None:
                    try:
                        sim.close()
                    except BaseException as exc:
                        cleanup_errors.append(_error_record(exc, "simulator_close", secrets()))
                        trace_error(exc, "simulator_close")
                summary = build_summary("pre_video_finalization")
                status = summary["status"]
                try:
                    write_artifacts(summary)
                    pre_video_written = True
                except Exception as exc:
                    cleanup_errors.append(_error_record(exc, "pre_video_summary", secrets()))
                    trace_error(exc, "pre_video_summary")
                state("finalizing_video")
                video_error = None
                if recorder is not None:
                    try:
                        recorder.close(status=status, error=json.dumps(error or cleanup_errors)
                                       if error or cleanup_errors else None)
                    except BaseException as exc:
                        video_error = _error_record(exc, "video_finalization", secrets())
                        trace_error(exc, "video_finalization")
                summary = build_summary("complete", video_error)
                status = summary["status"]
                write_artifacts(summary)
                with (output / "episodes.jsonl").open("w") as results:
                    _append(results, summary)
                state("finished")
                if console_pending is not None:
                    print(json.dumps({**console_pending, "final": True}), flush=True)
            finally:
                signals.restore()
    print(json.dumps({"event": "finished", **{k: summary[k] for k in (
        "status", "steps", "libero_success", "task_success", "v2_success", "api_calls", "control_without_video_s",
        "output")}, "physical_verdict": summary["physical_truth"].get("verdict"),
        "model_calls_by_kind": summary["model_calls"].get("calls_by_kind")}, default=str), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--store-dir", type=Path, help="ChatGPT subscription credential store (--live; as v1)")
    parser.add_argument("--profile", help="ChatGPT subscription credential profile (--live; as v1)")
    parser.add_argument("--task", choices=(*V2_TASK_KEYS, *V2_TASK_ALIASES), default=PRIMARY_TASK)
    parser.add_argument("--episode-index", type=int, choices=range(50), default=0)
    parser.add_argument("--expected-initial-observation-sha256",
                        help="Abort before inference if the reset observation differs (required for episodes "
                             "without a frozen registry hash)")
    parser.add_argument("--no-video", action="store_true", help="Skip the episode video")
    parser.add_argument("--smoke-script", choices=SMOKE_SCRIPTS)
    parser.add_argument("--smoke-steps", type=int,
                        help=f"Action cap for --sim-smoke (default {DEFAULT_SMOKE_STEPS} plumbing, "
                             f"{DEFAULT_RELATION_SMOKE_STEPS} relation)")
    parser.add_argument("--stop-after-steps", type=int,
                        help="--live only: an operator stop at the first complete-feedback boundary after N actions "
                             "(status user_stop; unset = no cap, the default protocol)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true",
                       help="Print the identity, prompt checks and v2 payloads; no credentials, simulator or files")
    modes.add_argument("--sim-smoke", action="store_true",
                       help="Real simulator, zero model calls, unscored (plumbing policy or the ORACLE relation run)")
    modes.add_argument("--live", action="store_true",
                       help="One scored episode with the ChatGPT subscription (real model calls)")
    args = parser.parse_args(argv)
    if (args.smoke_steps is not None or args.smoke_script is not None) and not args.sim_smoke:
        parser.error("--smoke-steps/--smoke-script require --sim-smoke")
    if args.stop_after_steps is not None and not args.live:
        parser.error("--stop-after-steps requires --live")
    smoke_script = args.smoke_script or "plumbing"
    if not args.live and (args.store_dir is not None or args.profile is not None):
        parser.error("only --live reads credentials; remove --store-dir/--profile")
    if not (args.preflight or args.sim_smoke or args.live):
        parser.error("choose a mode: --preflight, --sim-smoke (zero model calls) or --live (real model calls)")
    if not args.preflight and args.output is None:
        parser.error("--output is required for execution")
    config = configuration()
    try:
        sim_config, monitor = build_sim_config(args.task, sim_python=args.sim_python, libero_root=args.libero_root)
        _validate_expected_sha(args.expected_initial_observation_sha256)
        _validate_stop_after_steps(args.stop_after_steps)
        if args.sim_smoke:
            _validate_smoke_steps(args.smoke_steps if args.smoke_steps is not None else DEFAULT_SMOKE_STEPS)
        validate_configuration(config, sim_config, resolve_v2_task(args.task))
    except ValueError as exc:
        parser.error(str(exc))
    if args.preflight:
        try:
            result = preflight(config, sim_config, args.output, task_key=args.task, episode_index=args.episode_index,
                               libero_root=args.libero_root)
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, default=str), flush=True)
        return 0 if result["lint_check"]["passed"] else 2
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    try:
        check_render_environment(warn=lambda *a, **k: print(*a, file=sys.stderr, **k))
        resolve_expected_sha(resolve_v2_task(args.task).key, args.episode_index,
                             args.expected_initial_observation_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    if args.live:
        lint = lint_check_summary(args.libero_root, True)
        if not lint["passed"]:
            parser.error(f"T0 lint fails (require_all): missing {lint['missing']}, failing {lint['failing']}; a "
                         "live run needs a clean build (see --preflight)")
        try:                 # a lint-clean but broken client / plan build (zero calls, well under a second)
            live_build_check(resolve_v2_task(args.task).prompt)
        except Exception as exc:
            parser.error(f"live build check fails ({_live_build_failure(exc)}); a live run needs a working client "
                         "and plan build (see --preflight)")
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
    else:
        token_provider = _v1.offline_token_provider
    default_steps = DEFAULT_RELATION_SMOKE_STEPS if smoke_script == "relation" else DEFAULT_SMOKE_STEPS
    result = execute(args.output, config=config, sim_config=sim_config, task_key=args.task,
                     episode_index=args.episode_index, monitor_options=monitor, token_provider=token_provider,
                     expected_initial_observation_sha256=args.expected_initial_observation_sha256,
                     video=not args.no_video, smoke=bool(args.sim_smoke), smoke_script=smoke_script,
                     smoke_steps=args.smoke_steps if args.smoke_steps is not None else default_steps,
                     libero_root=args.libero_root, stop_after_steps=args.stop_after_steps)
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
