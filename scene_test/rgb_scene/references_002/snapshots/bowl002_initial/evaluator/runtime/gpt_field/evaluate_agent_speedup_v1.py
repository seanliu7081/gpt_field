"""Evaluate agent_SpeedUp_v1: model-supervised macro skills with sensor-grounded contact.

One fresh LIBERO episode (bowl_plate = LIBERO-Spatial task 0, drawer_bowl =
LIBERO-Goal task 3) at native 512 px, seed 7, under the speed-first evaluation
protocol: collisions are recorded only, raw LIBERO success is the only
completion, an operator stop file ends the run at a complete-feedback boundary,
and there is no action, call-count or wall-time horizon. gpt-6-astra (medium)
chooses macro skills and answers pointing requests; a local executor runs the
closed-loop primitives. --preflight prints the exact decision and pointing
payloads from a synthetic 512 observation without credentials, simulator or
files. --sim-smoke runs the real simulator against scripted offline envelopes
(zero model calls, unscored, capped at --smoke-steps). Helpers are reused from
scripts/evaluate_bowl_hybrid_subscription.py, which is loaded, never edited.

The exact stop command is printed at startup. A Ctrl+C or SIGTERM is a user stop; the simulator worker and
ffmpeg run in their own sessions, so only the runner receives it. During finalization further SIGINT/SIGTERM
are deferred (the third press hard-exits after the pre-video summary is on disk), and summary.json is written
before and after video finalization. --external-axes selects the policy's external-image mapping.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

import gpt_field.experiment as _experiment
from gpt_field.observation import decode_image
from gpt_field.ordered_transport import ordered_dumps


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = "scripts/evaluate_agent_speedup_v1.py"
HELPER_SCRIPTS = ("scripts/evaluate_bowl_hybrid_subscription.py", "scripts/evaluate_bowl_subscription_close.py",
                  "scripts/evaluate_bowl_field_history.py")
_spec = importlib.util.spec_from_file_location("_agent_speedup_v1_base_runner", ROOT / HELPER_SCRIPTS[0])
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def _helper(path):
    value = _runner
    for part in path.split("."):
        if not hasattr(value, part):
            raise ImportError(f"Runner helper {path} is missing from {HELPER_SCRIPTS[0]}")
        value = getattr(value, part)
    return value


configuration = _helper("configuration")
DEFAULT_SIM_PYTHON = _helper("DEFAULT_SIM_PYTHON")
DEFAULT_LIBERO_ROOT = _helper("DEFAULT_LIBERO_ROOT")
SimProcess = _helper("SimProcess")
LiberoConfig = _helper("LiberoConfig")
resolve_distance_task = _helper("resolve_distance_task")
DISTANCE_TASK_KEYS = _helper("DISTANCE_TASK_KEYS")
normalize_target_description = _helper("normalize_target_description")
_validate_episode_index = _helper("_validate_episode_index")
_validate_task_configuration = _helper("_validate_task_configuration")
_validate_collision_configuration = _helper("_validate_collision_configuration")
_collision_record = _helper("_collision_record")
termination_as_interrupt = _helper("_old._base.termination_as_interrupt")
_simulator_without_credentials = _helper("_old._simulator_without_credentials")
_safe_error = _helper("_old._safe_error")
_append = _helper("_old._base._append")
source_manifest = _helper("source_manifest")
digest = _helper("digest")
write_json = _helper("write_json")
APIError, MissingCredentials = _helper("APIError"), _helper("MissingCredentials")
BudgetExceeded, SchemaError = _helper("BudgetExceeded"), _helper("SchemaError")
PolicyObservation, image_to_data_url = _helper("PolicyObservation"), _helper("image_to_data_url")

METHOD = "agent_speedup_v1"
LIVE_MODE = "live_chatgpt_subscription_agent_speedup_v1"
SMOKE_MODE = "offline_scripted_agent_speedup_v1_sim_smoke"
OFFLINE_TOKEN = "offline-oauth-fixture"
OPERATOR_STOP_FILE = "operator_stop.request.json"
TRACEBACK_FILE = "error_traceback.txt"
DEFAULT_SMOKE_STEPS = 150
GRASP_SMOKE_STEPS = 420
MAX_SMOKE_STEPS = 2000
MAX_LOCAL_RETRIES = 5
IMAGE_SIZE = 512
STOPPING_RULE = ["LIBERO success", "operator stop file", "user stop",
                 "subscription authorization or quota error", "infrastructure error"]
RULINGS_ASSUMED = {"R-1_online_image_jacobian": True, "R-2_wrist_mount_facts": True,
                   "R-3_evaluator_tuned_constants": False, "R-4_episode_continuation_signal": False,
                   # live run 001 fix; --external-axes off disables it
                   "R-1x_external_image_axes_from_own_motion": True}
CONSTANT_LABELS = {
    "wrist_camera_offset_0.05m_along_R0": "ROBOT (R-2, robot.xml:239)",
    "grip_site_0.097m_along_R2": "ROBOT",
    "wrist_image_right_R1_up_R0_principal_point_centre": "ROBOT (R-2)",
    "gripper_max_opening_79mm_finger_gap_q1_minus_q2": "ROBOT",
    "image_jacobian_kappa_and_setpoint": "ONLINE (R-1)",
    "empty_close_curve": "ONLINE (Calibrate; recorded-run default only as fallback)",
    "touch_threshold_max_0.10mm_3sigma": "ONLINE",
    "stall_ratio_0.3x_free_motion_ratio": "ONLINE",
    "object_present_margin_0.5mm": "TUNED-initial",
    "hold_margin_1.0mm": "TUNED-initial",
    "align_tolerance_6px_gain_0.5_ncc_peak_0.6": "design default",
    "guarded_approach_10mm_command_150mm_bound": "design default",
    "r_bar_10mm_hold_stability_0.3mm": "TUNED-evaluator (R-3; not used)",
    "scene_geometry_object_poses": "ORACLE (never used by the policy)",
    # live run 001 fixes (agent_SpeedUp_v1.md §17.1); thresholds set from the run's own state8 traces
    "external_axes_3_probe_moves_4cm_farneback_flow": "ONLINE (R-1x, assumed)",
    "reach_limit_orientation_error_0.05rad_rising_3_steps": "design default (live 001: 53/61 failing moves, 1/17 false)",
    "uncontrolled_motion_20mm_per_step": "design default (free-space max 12.7 mm)",
    "held_gap_rise_1.0mm_or_0.4mm_step": "TUNED-initial (live 001 free carry <= 0.55 mm / 0.149 mm per step)",
    "place_gap_rise_0.3mm_lower_to_pick_minus_30mm": "TUNED-initial (live 001 free lowering drift <= 0.09 mm)",
    "orient_pose_drift_20mm_budget_tolerance_5deg": "design default (feasible 90 deg turn drifted 5.5 mm)",
    "align_probe_settle_0.03deg_0.3mm_2_steps_max_14": "design default (live 001 swing settled in 13-14 steps)",
}
PREFLIGHT_STATE = (0.0, 0.0, 1.0, math.pi, 0.0, 0.0, 0.04, -0.04)
# Frozen reset observations of the speed_first_v3 / operation_v4 protocol (512 px, seed 7, settle 10).
# Used when --expected-initial-observation-sha256 is omitted, so a changed reset fails closed.
DEFAULT_EXPECTED_SHA256 = {
    ("bowl_plate", 0): "8626d3e51bf8618a2c2793dbac9121dc8c87106197b95a7a6e7f483ffed65835",
    ("drawer_bowl", 0): "c6853a2d01f24753fce9fe3b06dd05f78d5d667c258fd048144c908f29321eb8",
}
RENDER_ENV_KEYS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES",
                   "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
SMOKE_SCRIPTS = ("plumbing", "grasp")
# Mirrors gpt_field.agent_speedup_v1.EXTERNAL_AXES_MODES (checked by the runner tests); kept here so the CLI
# and preflight do not depend on importing the policy module.
EXTERNAL_AXES_MODES = ("measured", "off")
EXTERNAL_AXES_HELP = ("measured: map external-image directions to world from the robot's own probe motion "
                      "(assumed R-1 extension); off: give the model no external mapping")
# Cleanup problems (simulator close, failed metadata reads, video finalization) are recorded separately and
# never override these episode outcomes; any other status becomes infrastructure_error.
OUTCOME_STATUSES = ("success", "user_stop", "diagnostic_complete", "diagnostic_sim_terminated")
FINALIZATION_ABORT_PRESSES = 3   # counting the Ctrl+C that stopped the episode
ERROR_DETAIL_CHARS = 500
EXECUTED_CONSOLE_EVERY = 25
STUCK_WINDOW, STUCK_FAILURES, STUCK_GRASPS_AFTER_LOSS = 12, 10, 5
SIM_CLOSE_DRAIN_S = 3.0
_BEARER = re.compile(r"(?i)(bearer\s+)[^\s'\",;]+")
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}")
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")


def offline_token_provider():
    """The scripted smoke's OAuth fixture; never a credential."""
    return OFFLINE_TOKEN


def _preflight_token_provider():
    raise MissingCredentials("Preflight must never read credentials")


def _no_network_transport(endpoint, payload, headers, timeout_s):
    raise APIError("Preflight must never send a request")


class NullRecorder:
    """--no-video recorder: accepts the recorder protocol and writes nothing."""

    def __init__(self, path=None, **kwargs):
        self.path = path

    def observe(self, observation, **kwargs):
        pass

    def decision(self, observation, decision, **kwargs):
        pass

    def executed(self, observation, **kwargs):
        pass

    def close(self, **kwargs):
        pass


def redact_secrets(text, secrets=()):
    for secret in secrets:
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "[REDACTED]")
    text = _BEARER.sub(r"\1[REDACTED]", text)
    return _API_KEY.sub("[REDACTED_KEY]", _JWT.sub("[REDACTED_JWT]", text))


def _write_traceback(output, exc, stage, secrets=()):
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with (Path(output) / TRACEBACK_FILE).open("a", encoding="utf-8") as stream:
        stream.write(f"== {type(exc).__name__} at stage {stage} ({datetime.now(timezone.utc).isoformat()}) ==\n")
        stream.write(redact_secrets(text, secrets))


def _error_record(exc, stage, secrets=()):
    """_safe_error's generic fields plus the redacted, truncated exception text (the real cause)."""
    return {**_safe_error(exc, stage), "detail": redact_secrets(str(exc), secrets)[:ERROR_DETAIL_CHARS]}


class _SessionSimProcess(SimProcess):
    """SimProcess whose worker runs in its own session, so a terminal Ctrl+C reaches only the runner.

    start() is SimProcess.start (gpt_field/libero.py) with ``start_new_session=True`` on the Popen; libero.py
    is not edited. The worker exits on stdin EOF. close() first drains any in-flight response, so a worker that
    was mid-step when the runner stopped can write its reply, read EOF and exit, instead of blocking on a full
    pipe until SimProcess.close() times out and terminates it."""

    def start(self):
        if self._process is not None:
            return self
        command = self._command or [
            self.config.python_executable or sys.executable, "-u", "-m", "gpt_field.libero",
            "--worker", "--config-json", json.dumps(asdict(self.config)),
        ]
        env = os.environ.copy()
        package_src = str(Path(sys.modules[SimProcess.__module__].__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_src, env.get("PYTHONPATH")]))
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0,
                                         start_new_session=True)
        try:
            self.metadata = self._request({"op": "metadata"})
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        proc = self._process
        if proc is not None and proc.stdin is not None and proc.stdout is not None:
            try:
                proc.stdin.close()
                deadline = time.monotonic() + SIM_CLOSE_DRAIN_S
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while (remaining := deadline - time.monotonic()) > 0 and selector.select(remaining):
                        if not os.read(proc.stdout.fileno(), 1 << 20):
                            break
            except (OSError, ValueError):
                pass
        super().close()


def _hard_exit(code):
    os._exit(code)


class _FinalizationSignals:
    """Defer SIGINT/SIGTERM while the runner writes its terminal artifacts.

    Presses are counted, including the Ctrl+C that stopped the episode. Each deferred press prints how many more
    abort; the FINALIZATION_ABORT_PRESSES-th runs ``on_abort`` (which puts the pre-video summary on disk) and
    hard-exits with 130, so a wedged encoder cannot hang the runner. Main thread only (signal.signal)."""

    def __init__(self, presses=0, on_abort=None, stream=None):
        self.presses = presses
        self.deferred = []
        self.on_abort = on_abort
        self.stream = stream
        self._previous = {}

    def _print(self, text):
        print(text, file=self.stream or sys.stderr, flush=True)

    def _notice(self):
        remaining = FINALIZATION_ABORT_PRESSES - self.presses
        self._print(f"finalizing: writing summary/video; press Ctrl+C {remaining} more "
                    f"time{'' if remaining == 1 else 's'} to abort")

    def install(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.signal(signum, self._deferred)
            except (ValueError, OSError):
                pass
        if self.presses:
            self._notice()
        return self

    def _deferred(self, signum, _frame):
        self.presses += 1
        self.deferred.append(signal.Signals(signum).name)
        if self.presses < FINALIZATION_ABORT_PRESSES:
            self._notice()
            return
        self._print("finalizing aborted by operator: ensuring the pre-video summary is on disk, then exiting (130)")
        try:
            if self.on_abort is not None:
                self.on_abort()
        except BaseException as exc:  # never block the hard exit
            self._print(f"pre-video summary could not be written: {type(exc).__name__}")
        finally:
            _hard_exit(130)

    def restore(self):
        for signum, handler in self._previous.items():
            if handler is None:
                continue
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, TypeError):
                pass
        self._previous.clear()


def _stuck_reason(outcomes):
    """Operator-facing stuck heuristic (alert only; the protocol has no automatic cap)."""
    window = outcomes[-STUCK_WINDOW:]
    failing = sum(item.get("status") in ("failed", "rejected") for item in window)
    if failing >= STUCK_FAILURES:
        return f"{failing} of the last {len(window)} macro outcomes failed or were rejected"
    lost = max((index for index, item in enumerate(outcomes) if item.get("reason") == "lost_in_transport"), default=None)
    if lost is None:
        return None
    after = outcomes[lost + 1:]
    if any(item.get("skill") == "grasp_with" and item.get("status") == "succeeded" for item in after):
        return None
    grasps = sum(item.get("skill") == "grasp_with" and item.get("status") == "failed" for item in after)
    if grasps > STUCK_GRASPS_AFTER_LOSS:
        return f"lost_in_transport (outcome n={outcomes[lost].get('n')}) followed by {grasps} failed grasp_with outcomes"
    return None


def _sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def _qualname(value):
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _optional_module(name, errors=None):
    """Import gpt_field.<name> for metadata; None (with the reason recorded) if it cannot be imported."""
    try:
        return importlib.import_module(f"gpt_field.{name}")
    except ImportError as exc:
        if errors is not None:
            errors[name] = f"{type(exc).__name__}: {exc}"
        return None


def _checked_json(value):
    json.dumps(value, allow_nan=False)
    return value


def _validate_local_retries(value):
    if type(value) is not int or not 0 <= value <= MAX_LOCAL_RETRIES:
        raise ValueError(f"local_retries must be an integer in [0, {MAX_LOCAL_RETRIES}]")
    return value


def _validate_external_axes(value):
    if value not in EXTERNAL_AXES_MODES:
        raise ValueError(f"external_axes must be one of {EXTERNAL_AXES_MODES}")
    return value


def _validate_smoke_steps(value):
    if type(value) is not int or not 1 <= value <= MAX_SMOKE_STEPS:
        raise ValueError(f"smoke_steps must be an integer in [1, {MAX_SMOKE_STEPS}]")
    return value


def _validate_expected_sha(value):
    if value is not None and (type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)):
        raise ValueError("expected_initial_observation_sha256 must be 64 lowercase hex characters")
    return value


def build_sim_config(task_key, *, sim_python=DEFAULT_SIM_PYTHON, libero_root=DEFAULT_LIBERO_ROOT):
    """Native 512, seed 7, no horizon, record-only task_manipulation_v1 collisions."""
    task_spec = resolve_distance_task(task_key)
    return LiberoConfig(suite=task_spec.suite, seed=7, max_steps=None,
                        libero_root=str(Path(libero_root).expanduser().resolve()),
                        python_executable=os.path.abspath(os.path.expanduser(sim_python)),
                        include_arm_joints=False, render_size=IMAGE_SIZE, image_size=IMAGE_SIZE,
                        collision_check_mode="task_manipulation_v1", stop_on_collision=False)


def validate_configuration(config, sim_config, task_spec):
    _validate_task_configuration(config, sim_config, task_spec, speed_first_v3=True)
    _validate_collision_configuration(sim_config, False, speed_first=True)
    if sim_config.include_arm_joints:
        raise ValueError("agent_speedup_v1 uses the state8 protocol; include_arm_joints must be False")
    if sim_config.initial_state_override is not None or sim_config.task_bddl_override is not None:
        raise ValueError("agent_speedup_v1 evaluates official LIBERO scenes only; overrides are unsupported")
    if not sim_config.stop_on_success:
        raise ValueError("LIBERO success must stop the episode")


def _module_constant(modules, name):
    for module in modules:
        if module is not None and getattr(module, name, None) is not None:
            return getattr(module, name)
    return None


def agent_metadata(task_key, target_description=None, *, local_retries=2, video=True):
    """Identity block: module versions, prompt/schema hashes, rulings and constant labels."""
    errors = {}
    client_module = _optional_module("agent_speedup_v1_client", errors)
    policy_module = _optional_module("agent_speedup_v1", errors)
    names = {"AGENT_SPEEDUP_V1_VERSION": (client_module, policy_module),
             "AGENT_PROMPT_VERSION": (client_module, policy_module),
             "HISTORY_PROTOCOL": (client_module,),
             "POLICY_VERSION": (policy_module,),
             "MARK_PROTOCOL_VERSION": (_optional_module("mark_prompting", errors),),
             "ORDERED_TRANSPORT_VERSION": (_optional_module("ordered_transport", errors),),
             "CONTACT_SKILLS_VERSION": (_optional_module("contact_skills", errors),),
             "WRIST_SERVO_VERSION": (_optional_module("wrist_servo", errors),),
             "AGENT_VIDEO_VERSION": (_optional_module("agent_speedup_v1_video", errors),)}
    versions = {name: _module_constant(modules, name) for name, modules in names.items()}
    prompts = {"decision_instructions_sha256": None, "decision_schema_sha256": None,
               "decision_schema_property_order": None,
               "pointing_instructions_sha256": {"coarse": None, "fine": None},
               "pointing_schema_sha256": {"coarse": None, "fine": None},
               "hash_serialization": "sha256 of UTF-8 text; schemas via ordered_dumps (wire insertion order)"}
    if client_module is not None:
        if callable(getattr(client_module, "decision_instructions", None)):
            prompts["decision_instructions_sha256"] = _sha256(client_module.decision_instructions(task_key, target_description))
        if callable(getattr(client_module, "decision_schema", None)):
            schema = client_module.decision_schema(task_key)
            prompts["decision_schema_sha256"] = _sha256(ordered_dumps(schema))
            prompts["decision_schema_property_order"] = list(schema.get("properties", {}))
        for stage in ("coarse", "fine"):
            if callable(getattr(client_module, "pointing_instructions", None)):
                prompts["pointing_instructions_sha256"][stage] = _sha256(client_module.pointing_instructions(stage))
            if callable(getattr(client_module, "pointing_schema", None)):
                prompts["pointing_schema_sha256"][stage] = _sha256(ordered_dumps(client_module.pointing_schema(stage)))
    labels = getattr(policy_module, "CONSTANT_LABELS", None)
    policy_rulings = getattr(policy_module, "RULINGS_ASSUMED", None)
    return {"versions": versions, **prompts, "rulings_assumed": dict(RULINGS_ASSUMED),
            "policy_rulings_assumed": dict(policy_rulings) if isinstance(policy_rulings, dict) else None,
            "constant_labels": dict(labels) if isinstance(labels, dict) else dict(CONSTANT_LABELS),
            "local_retries": local_retries, "video": video, "target_description": target_description,
            "model_role": "macro skill supervisor and numbered-mark pointing; no metric contact quantities",
            "contact_quantities_from_model": False, "gripper_close_gate": "sensor-gated inside grasp_with only",
            "request_history": "stateless (current images, state summary, outcome ledger)",
            "image_orientation": "client-side horizontal flip to upright; worker, traces and hashes stay legacy",
            "model_declares_completion": False, "field_or_direct_controllers": None,
            "module_import_errors": errors}


def _client_identity(client, config, target_description):
    stats = client.history_stats()
    return {"class": _qualname(client), "model": config.model, "prompt_version": client.prompt_version,
            "reasoning_effort": client.reasoning_effort, "history_protocol": stats.get("protocol"),
            "initial_history_settings": stats, "task_prompt": getattr(client, "task_prompt", None),
            "target_description": target_description}


def _default_client_factory(smoke):
    from gpt_field.agent_speedup_v1_client import AgentSpeedUpV1Client
    if not smoke:
        return AgentSpeedUpV1Client

    class AgentSpeedUpV1SmokeClient(AgentSpeedUpV1Client):
        """Unchanged client fed scripted offline envelopes; never a live request."""

        def budget_snapshot(self):
            usage = super().budget_snapshot()
            return {**usage, "billing": "offline_mock", "api_calls": 0, "subscription_inference_calls": 0,
                    "mock_response_calls": usage.get("api_calls", 0)}

        def _write_log(self, path, data):
            if isinstance(data, dict):
                data = {**data, "offline_smoke": True, "offline_billing": "offline_mock",
                        "scripted_transport": "ScriptedAgentTransport"}
            super()._write_log(path, data)

    return AgentSpeedUpV1SmokeClient


def _default_policy_factory(client, *, local_retries=2, external_axes="measured"):
    from gpt_field.agent_speedup_v1 import AgentSpeedUpV1Policy
    return AgentSpeedUpV1Policy(client, local_retries=local_retries, external_axes=external_axes)


def _build_policy(policy_factory, client, *, local_retries, external_axes):
    """external_axes goes to the default factory (the real policy); injected test factories keep the old
    (client, *, local_retries) contract. A policy that reports a different external_axes fails closed."""
    if policy_factory is None:
        policy = _default_policy_factory(client, local_retries=local_retries, external_axes=external_axes)
    else:
        policy = policy_factory(client, local_retries=local_retries)
    reported = getattr(policy, "external_axes", None)
    if reported is not None and reported != external_axes:
        raise ValueError(f"policy external_axes {reported!r} differs from the requested {external_axes!r}")
    return policy


def _default_recorder_factory():
    from gpt_field.agent_speedup_v1_video import AgentSpeedUpV1VideoRecorder
    return AgentSpeedUpV1VideoRecorder


def _scripted_transport(task_key, script="plumbing"):
    from gpt_field.agent_speedup_v1_client import ScriptedAgentTransport, default_smoke_script
    if script == "plumbing":
        return ScriptedAgentTransport(default_smoke_script(task_key))
    if script == "grasp" and task_key == "bowl_plate":
        # Verified on real LIBERO (bowl_plate init 0, start pose): coarse 18 / fine 20 is the right rim of the
        # target bowl in the upright wrist view. Exercises pointing, Align, ladder, close and the hold test.
        null = {"target": None, "strategy": None, "side": None, "approach": None,
                "waypoint_offset_m": None, "max_travel_m": None}
        return ScriptedAgentTransport([
            {"situation": "Scripted smoke: rim pinch on the target bowl's right rim.", "skill": "grasp_with",
             "params": {**null, "target": "black_bowl", "strategy": "rim_pinch", "side": "wrist_right"}},
            {"situation": "Scripted smoke: back away.", "skill": "retreat", "params": dict(null)}],
            coarse=(18,), fine=(20,))
    raise ValueError("the grasp smoke script is defined only for bowl_plate episode 0")


def render_environment():
    return {key: os.environ.get(key) for key in RENDER_ENV_KEYS}


def check_render_environment(warn=print):
    env = render_environment()
    if env["CUDA_VISIBLE_DEVICES"] and env["MUJOCO_EGL_DEVICE_ID"]:
        raise ValueError("unset CUDA_VISIBLE_DEVICES when MUJOCO_EGL_DEVICE_ID is set (robosuite asserts one "
                         "is a substring of the other; the GPU minor order is reversed on this host)")
    if not env["MUJOCO_EGL_DEVICE_ID"]:
        warn("warning: MUJOCO_EGL_DEVICE_ID is unset; rendering will use EGL device 0", flush=True)
    return env


def _require_offline_transport(transport):
    if getattr(transport, "offline_mock", False) is True:
        return transport
    module = _optional_module("agent_speedup_v1_client")
    if module is not None and isinstance(transport, module.ScriptedAgentTransport):
        return transport
    raise ValueError("Scripted sim smoke accepts only an offline scripted transport")


# preflight -----------------------------------------------------------------
def _image_placeholder(value):
    return f"<{value.split(',', 1)[0]} {len(value)} chars sha256={_sha256(value)}>"


def _payload_preview(value):
    if isinstance(value, str) and value.startswith("data:image/"):
        return _image_placeholder(value)
    if isinstance(value, dict):
        return {key: _payload_preview(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload_preview(item) for item in value]
    return value


def _rotation_matrix(rotvec):
    from gpt_field.robot_tool_geometry import _rotation_matrix as rotation
    return rotation(np.asarray(rotvec, dtype=float))


def synthetic_state_summary(state):
    """Preview-only summary from state8; the live policy builds its own."""
    R = _rotation_matrix(state[3:6])
    vector = lambda values: [round(float(v), 4) + 0.0 for v in values]
    return {"eef_xyz_m": vector(state[:3]),
            "tool_axes_world": {"jaw_R1": vector(R[:, 1]), "approach_R2": vector(R[:, 2]),
                                "wrist_image_right": vector(R[:, 1]), "wrist_image_up": vector(R[:, 0])},
            "finger_gap_mm": round((state[6] - state[7]) * 1000, 2), "ledger": [], "last_outcome": None,
            "synthetic_preview": True}


def _pointing_images(upright_rgb, upright_url):
    marks = _optional_module("mark_prompting")
    if marks is None:
        return upright_url, upright_url, None, "unavailable: gpt_field.mark_prompting is not importable"
    fine_url, origin = marks.fine_overlay(upright_rgb, 36)
    return marks.coarse_overlay(upright_rgb), fine_url, [int(v) for v in origin], marks.MARK_PROTOCOL_VERSION


def preflight(config, sim_config, output=None, *, task_key="bowl_plate", episode_index=0,
              target_description=None, local_retries=2, video=True, client_factory=None,
              external_axes="measured"):
    """Exact decision and pointing payloads for a synthetic 512 observation; no auth, network, files or sim."""
    episode_index = _validate_episode_index(episode_index)
    task_spec = resolve_distance_task(task_key)
    target_description = normalize_target_description(target_description)
    local_retries = _validate_local_retries(local_retries)
    external_axes = _validate_external_axes(external_axes)
    validate_configuration(config, sim_config, task_spec)
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh agent_speedup_v1 experiment requires a new output directory")
    client = (client_factory or _default_client_factory(False))(
        config, token_provider=_preflight_token_provider, task_key=task_spec.key,
        target_description=target_description, transport=_no_network_transport, log_dir=None)
    size = sim_config.image_size
    legacy = image_to_data_url(np.zeros((size, size, 3), dtype=np.uint8), size=size, rotate=False)
    observation = PolicyObservation(legacy, legacy, PREFLIGHT_STATE, task_spec.prompt)
    upright_rgb = np.ascontiguousarray(np.asarray(decode_image(observation.wrist_image))[:, ::-1])
    upright = image_to_data_url(upright_rgb, size=size, rotate=False)
    from gpt_field import agent_speedup_v1 as agent_policy
    from gpt_field.contact_skills import StepView
    try:
        state_summary = agent_policy.AgentSpeedUpV1Policy(
            client, local_retries=local_retries, external_axes=external_axes)._state_summary(StepView(observation))
        summary_source = "the live policy's own _state_summary on the synthetic observation (no calls)"
    except Exception as exc:  # fakes in tests may not support the full policy
        state_summary = synthetic_state_summary(observation.state)
        summary_source = f"runner synthetic preview ({type(exc).__name__})"
    decision_payload = client.build_payload({"state_summary": state_summary,
                                             "external_image": upright, "wrist_image": upright}, mode="decision")
    coarse_url, fine_url, fine_origin, mark_protocol = _pointing_images(upright_rgb, upright)
    if task_spec.key == "drawer_bowl":
        phrase = agent_policy.bar_target_phrase("top_drawer_handle")
        context = agent_policy.bar_pointing_context("top_drawer_handle", "world_-y")
    else:
        phrase = agent_policy.rim_target_phrase("black_bowl", "wrist_left")
        context = agent_policy.rim_pointing_context("black_bowl", "wrist_left", target_description)
    coarse = client.build_payload({"stage": "coarse", "image": coarse_url, "target_phrase": phrase,
                                   "context": dict(context)}, mode="pointing_coarse")
    fine = client.build_payload({"stage": "fine", "image": fine_url, "target_phrase": phrase,
                                 "context": {**context, "cell_id": 36}}, mode="pointing_fine")
    return {"status": "preflight_only", "method": METHOD, "task": task_spec.prompt, "task_id": task_spec.task_id,
            "distance_task": task_spec.key, "episode_index": episode_index, "billing": "chatgpt_subscription",
            "simulator": asdict(sim_config), "api": asdict(config),
            "client": _client_identity(client, config, target_description),
            "agent_speedup_v1": agent_metadata(task_spec.key, target_description, local_retries=local_retries, video=video),
            "synthetic_observation": True, "synthetic_state": list(observation.state),
            "state_summary_source": summary_source, "render_environment": render_environment(),
            "default_expected_initial_observation_sha256": DEFAULT_EXPECTED_SHA256.get((task_spec.key, episode_index)),
            "mark_protocol": mark_protocol, "api_calls": 0, "api_key_calls": 0, "credentials_checked": False,
            "simulator_started": False, "files_written": False,
            "sim_python_exists": Path(sim_config.python_executable or "").is_file(),
            "libero_checkout_exists": (Path(sim_config.libero_root or "") / "libero/libero/benchmark/__init__.py").is_file(),
            "payload": _payload_preview(decision_payload),
            "pointing_payloads": {"coarse": _payload_preview(coarse), "fine": _payload_preview(fine)},
            "stopping_rule": STOPPING_RULE, "evaluation_scored": True, "video": video, "local_retries": local_retries,
            "external_axes": external_axes, "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
            "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False}


# execution -----------------------------------------------------------------
def _read_sources(attempts=3):
    """source_manifest() plus the exact hashed bytes; retried if a file changes while being read."""
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
    scripts = {RUNNER_PATH: Path(__file__).read_bytes(),
               **{path: (ROOT / path).read_bytes() for path in HELPER_SCRIPTS}}
    return software, sources, scripts


def _attribute(owner, name):
    value = getattr(owner, name)
    return value() if callable(value) else value


def execute(output, *, config, sim_config, task_key, episode_index, token_provider=None, transport=None,
            client_factory=None, policy_factory=None, sim_factory=None, recorder_factory=None,
            expected_initial_observation_sha256=None, target_description=None, local_retries=2,
            video=True, smoke=False, smoke_steps=DEFAULT_SMOKE_STEPS, smoke_script="plumbing",
            external_axes="measured"):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError("Fresh agent_speedup_v1 experiment requires a new output directory; no resume")
    episode_index = _validate_episode_index(episode_index)
    task_spec = resolve_distance_task(task_key)
    target_description = normalize_target_description(target_description)
    local_retries = _validate_local_retries(local_retries)
    external_axes = _validate_external_axes(external_axes)
    expected = _validate_expected_sha(expected_initial_observation_sha256)
    expected_source = "argument" if expected is not None else None
    if expected is None and (task_key, episode_index) in DEFAULT_EXPECTED_SHA256 and sim_factory is None:
        expected, expected_source = DEFAULT_EXPECTED_SHA256[(task_key, episode_index)], "default_frozen_reset"
    if smoke_script not in SMOKE_SCRIPTS:
        raise ValueError(f"smoke_script must be one of {SMOKE_SCRIPTS}")
    if type(video) is not bool or type(smoke) is not bool:
        raise ValueError("video and smoke must be booleans")
    smoke_steps = _validate_smoke_steps(smoke_steps) if smoke else None
    validate_configuration(config, sim_config, task_spec)
    if smoke:
        if token_provider not in (None, offline_token_provider):
            raise ValueError("Scripted sim smoke uses only its offline OAuth fixture, never credentials")
        token_provider = offline_token_provider
        transport = (_scripted_transport(task_spec.key, smoke_script) if transport is None
                     else _require_offline_transport(transport))
    elif not callable(token_provider):
        raise MissingCredentials("Live agent_speedup_v1 execution requires a ChatGPT subscription token provider")
    client = (client_factory or _default_client_factory(smoke))(
        config, token_provider=token_provider, task_key=task_spec.key, target_description=target_description,
        transport=transport, log_dir=output / "requests")
    policy = _build_policy(policy_factory, client, local_retries=local_retries, external_axes=external_axes)
    worker_session = "own_session" if sim_factory is None else "injected_sim_factory"
    sim_factory = sim_factory or _SessionSimProcess
    recorder_factory = recorder_factory or (_default_recorder_factory() if video else NullRecorder)
    software, sources, scripts = _read_sources()
    mode = SMOKE_MODE if smoke else LIVE_MODE
    billing = "offline_mock" if smoke else "chatgpt_subscription"
    identity = {"kind": "agent_speedup_v1_sim_smoke" if smoke else "agent_speedup_v1_full_task",
        "mode": mode, "method": METHOD, "task": task_spec.prompt, "task_id": task_spec.task_id,
        "episode_index": episode_index, "distance_task": task_spec.key, "task_spec": asdict(task_spec),
        "simulator": asdict(sim_config), "api": asdict(config), "billing": billing, "software": software,
        "runner": {"path": RUNNER_PATH, "sha256": _sha256(scripts[RUNNER_PATH])},
        "runner_helpers": [{"path": path, "sha256": _sha256(scripts[path])} for path in HELPER_SCRIPTS],
        "client": _client_identity(client, config, target_description),
        "policy": {"class": _qualname(policy), "method": METHOD, "controller": "agent_skill",
                   "controllers": ["agent_skill"], "local_retries": local_retries, "external_axes": external_axes,
                   "update_steps": 1,
                   "grid": None, "field_representation": None, "field_update_steps": None,
                   "field_execution_steps": None, "direct_action_steps": None, "planner": None,
                   "routing_source": "model macro-skill decisions from upright RGB and a state summary; "
                                     "local sensor-gated primitives",
                   "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
                   "object_state_feedback_to_policy": False},
        "agent_speedup_v1": agent_metadata(task_spec.key, target_description, local_retries=local_retries, video=video),
        "zero_shot": True, "collision_policy": "record_only", "success_qualification": "raw_libero_goal",
        "primary_timing_metric": "control_without_video_s", "operator_stop_file": OPERATOR_STOP_FILE,
        "camera_render_size": IMAGE_SIZE, "policy_image_size": IMAGE_SIZE,
        "expected_initial_observation_sha256": expected, "expected_initial_observation_sha256_source": expected_source,
        "video": video, "local_retries": local_retries, "external_axes": external_axes,
        "render_environment": render_environment(), "simulator_worker_session": worker_session,
        "smoke_steps": smoke_steps, "smoke_script": smoke_script if smoke else None,
        "stopping_rule": [f"{smoke_steps} scripted smoke actions", *STOPPING_RULE] if smoke else STOPPING_RULE,
        "stop_on_model_complete": False, "model_completion_is_libero_success": False,
        "max_policy_steps": smoke_steps, "max_api_calls": None, "max_wall_time_s": None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "evaluation_scored": not smoke, "prior_budgets_inherited": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    frozen = output / "frozen_source"
    frozen.mkdir()
    for name, data in sources.items():
        (frozen / name).write_bytes(data)
    for path, data in scripts.items():
        (frozen / Path(path).name).write_bytes(data)
    trace_path = output / "trace.jsonl"
    stop_command = f"touch {shlex.quote(str(output / OPERATOR_STOP_FILE))}"
    print(f"operator stop (ends the run at the next complete-feedback boundary): {stop_command}",
          file=sys.stderr, flush=True)

    status, stage, error = "running", "starting", None
    caught, interrupted_stage, stop_source = None, None, None
    steps = refreshes = pointing_count = feedback_count = close_actions = open_actions = neutral_actions = 0
    pending_action, first_close_event, success_seen, path_length = None, None, False, 0.0
    controller = macro = primitive = None
    outcomes = []
    recorder, sim, cleanup_errors, failed_reads = None, None, [], {}
    started = time.monotonic()
    control_started = control_stopped = None
    collision = None
    timings = {key: 0.0 for key in ("decision_s", "model_pipeline_s", "simulator_step_s",
                                    "observe_transition_s", "video_render_s")}
    stuck_alert, stuck_alerts, stuck_armed, decisions_seen, console_pending = None, [], True, 0, None
    collected, pre_video_written, signals = {}, False, None
    secrets = lambda: list(getattr(client, "_used_access_tokens", ()) or ())

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

    def record_decision_time(latency, details):
        timings["decision_s"] += latency
        timings["model_pipeline_s"] += float(details.get("api_latency_s") or 0.0)

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json(output / "run_state.json", {"status": status, "stage": stage, "mode": mode, "method": METHOD,
            "task_id": task_spec.task_id, "episode_index": episode_index, "step": steps,
            "controller": controller, "macro": macro, "primitive": primitive,
            "pending_action_index": pending_action, "skill_decision_acts": refreshes,
            "pointing_results": pointing_count, "macro_outcomes": len(outcomes),
            "budget": guarded("budget_snapshot", client.budget_snapshot),
            "history_stats": guarded("history_stats", client.history_stats),
            "agent": guarded("agent_stats", policy.agent_stats),
            "collision": collision, "first_close_event": first_close_event, "error": error,
            "stuck_alert": stuck_alert, "operator_stop_command": stop_command})

    # operator console (R3): one compact line per decision and per macro outcome, plus a stuck alert ---------
    def wall_s():
        return round(time.monotonic() - started, 1)

    def api_calls():
        return (guarded("budget_snapshot", client.budget_snapshot, {}) or {}).get("api_calls")

    def console(row):
        print(json.dumps(row, separators=(",", ":"), default=str), flush=True)

    def note_decision(routing, superseded):
        nonlocal decisions_seen
        decisions_seen += 1
        routing = routing if isinstance(routing, dict) else {}
        params = routing.get("params")
        console({"event": "decision", "k": decisions_seen, "skill": routing.get("skill"),
                 "params": ({key: value for key, value in params.items() if value is not None}
                            if isinstance(params, dict) else params),
                 "superseded_in_act": superseded, "step": steps, "wall_s": wall_s(), "api_calls": api_calls()})

    def note_outcome(record):
        nonlocal stuck_alert, stuck_armed
        console({"event": "outcome", **{key: record.get(key) for key in ("n", "skill", "status", "reason", "step")},
                 "wall_s": wall_s(), "api_calls": api_calls()})
        if record.get("status") == "succeeded":
            if not stuck_armed and stuck_alert is not None:
                stuck_alert.update(active=False, cleared_by_outcome_n=record.get("n"), cleared_at_step=steps)
            stuck_armed = True
            return
        reason = _stuck_reason(outcomes) if stuck_armed else None
        if reason is None:
            return
        stuck_armed = False
        stuck_alert = {"active": True, "reason": reason, "outcome_n": record.get("n"), "step": steps,
                       "wall_s": wall_s(), "api_calls": api_calls(), "stop_command": stop_command,
                       "automatic_cap": False}
        stuck_alerts.append(stuck_alert)
        print(f"STUCK ALERT: {reason} | step {steps} | wall {stuck_alert['wall_s']:.1f} s | api calls "
              f"{stuck_alert['api_calls']} | the run continues (no automatic cap); to stop at the next "
              f"complete-feedback boundary: {stop_command}", file=sys.stderr, flush=True)
        state(stage)

    # finalization (R1, R7a, R8) --------------------------------------------------------------------------
    def finalization_trace_events():
        """Outcomes the policy finished in an act the runner never logged (the act an interrupt aborted), and
        the operator_stop event of a Ctrl+C / SIGTERM stop, so both stop paths look the same in the trace."""
        known = {item.get("n") for item in outcomes if item.get("n") is not None}
        added, events = [], []
        for outcome in list(getattr(policy, "outcomes", None) or []):
            n = outcome.get("n") if isinstance(outcome, dict) else None
            if n is None or n in known:
                continue
            known.add(n)
            record = {"n": n, "step": outcome.get("step"), "skill": outcome.get("skill", outcome.get("macro")),
                      "status": outcome.get("status"), "reason": outcome.get("reason"),
                      "recorded_at_finalization": True}
            outcomes.append(record)
            added.append(record)
            events.append({"event": "macro_outcome", "step": outcome.get("step"), "outcome": outcome, "mode": mode,
                           "recorded_at_finalization": True})
        if isinstance(caught, KeyboardInterrupt):
            events.append({"event": "operator_stop", "source": stop_source, "step": steps,
                           "feedback_count": feedback_count, "pending_action_index": pending_action,
                           "request_file": None, "stage": interrupted_stage})
        if events and trace_path.is_file():
            with trace_path.open("a") as log:
                for row in events:
                    _append(log, row)
        return added

    def collect():
        if not collected:
            collected.update({
                "agent_stats": guarded("agent_stats", policy.agent_stats),
                "ledger": guarded("ledger", lambda: list(_attribute(policy, "ledger"))),
                "calibration": guarded("calibration", lambda: (None if _attribute(policy, "calibration") is None
                                                              else _attribute(policy, "calibration").to_dict())),
                "budget": guarded("budget_snapshot", client.budget_snapshot, {}) or {},
                "history": guarded("history_stats", client.history_stats, {}) or {}})
        return collected

    def outcome_status(video_error=None):
        if (cleanup_errors or failed_reads or video_error) and status not in OUTCOME_STATUSES:
            return "infrastructure_error"
        return status

    def build_summary(finalization_stage, video_error=None):
        data = collect()
        ledger, budget, history = data["ledger"], data["budget"], data["history"]
        final = outcome_status(video_error)
        completed = not smoke and final == "success"
        return {"status": final, "mode": mode, "method": METHOD, "task": task_spec.prompt,
            "suite": task_spec.suite, "task_id": task_spec.task_id, "distance_task": task_spec.key,
            "episode_index": episode_index, "seed": 7, "requested_episodes": 1, "steps": steps,
            "skill_decision_acts": refreshes, "pointing_results": pointing_count,
            "calls_by_kind": history.get("calls_by_kind"), "macro_outcomes": len(outcomes),
            "macro_outcome_counts": dict(Counter(f"{item['skill']}:{item['status']}" for item in outcomes)),
            "macro_outcome_sequence": list(outcomes),
            "outcomes_recorded_at_finalization": sum(bool(item.get("recorded_at_finalization")) for item in outcomes),
            "final_controller": controller, "final_macro": macro,
            "final_primitive": primitive, "measured_feedback_count": feedback_count,
            "gripper_close_actions": close_actions, "gripper_open_actions": open_actions,
            "gripper_neutral_actions": neutral_actions, "path_length_m": path_length,
            "pending_action_index": pending_action, "first_close_event": first_close_event,
            "local_retries": local_retries, "external_axes": external_axes, "calibration": data["calibration"],
            "ledger_entries": len(ledger) if isinstance(ledger, list) else None,
            "agent_stats": "agent_stats.json", "ledger": "ledger.json",
            "diagnostic_complete": smoke and final in ("diagnostic_complete", "diagnostic_sim_terminated"),
            "smoke_steps": smoke_steps, "stop_on_model_complete": False, "model_declared_complete": False,
            "model_completion": None, "benchmark_completed": completed,
            "reasoning_effort": getattr(client, "reasoning_effort", None),
            "prompt_version": getattr(client, "prompt_version", None),
            "timing_components": dict(timings),
            "timing_components_note": "model_pipeline_s (sum of api_latency_s over decision and pointing calls) is "
                "included in decision_s; observe_transition_s, simulator_step_s and video_render_s are separate. "
                "control_without_video_s = control_elapsed_s - video_render_s. Terminal video and cleanup are "
                "excluded from control_elapsed_s.",
            "libero_success_observed": success_seen, "libero_success": terminal_raw_libero_success,
            "task_success": completed, "collision": collision, "completion_time_eligible": completed,
            "failure_reason": None, "evaluation_completed": completed,
            "success_rate": 1.0 if completed else None, "scored_episodes": int(completed),
            "incomplete_episodes": int(not completed), "evaluation_scored": not smoke,
            "api_calls": budget.get("api_calls"),
            "mock_response_calls": budget.get("mock_response_calls", 0) if smoke else 0,
            "billing": billing, "budget": budget, "history_stats": history,
            "api_key_calls": budget.get("api_key_calls", 0), "api_key_fallback": False,
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
            "finalization": {"stage": finalization_stage,
                             "deferred_signals": list(signals.deferred) if signals is not None else []},
            "output": str(output),
            "interpretation": ("Scripted agent_speedup_v1 smoke with real simulator execution and offline "
                "envelopes; no model calls and no performance score." if smoke else
                "One zero-shot agent_speedup_v1 episode. Only raw LIBERO success completes the evaluation. "
                "Collisions are evaluator-only secondary metrics. Video-excluded elapsed time is a completion "
                "time only when completion_time_eligible is true. Other stops remain incomplete; no automatic "
                "episode cap.")}

    def write_artifacts(summary):
        data = collect()
        write_json(output / "runner_summary.json", summary)
        write_json(output / "summary.json", summary)
        if collision is not None:
            write_json(output / "collision.json", collision)
        write_json(output / "task_evaluation.json", {key: summary[key] for key in (
            "status", "task_success", "libero_success", "collision", "failure_reason",
            "evaluation_completed", "success_rate", "scored_episodes", "incomplete_episodes",
            "completion_time_eligible")})
        write_json(output / "agent_stats.json", data["agent_stats"])
        write_json(output / "ledger.json", data["ledger"])

    def abort_finalization():
        """Third Ctrl+C during finalization: the pre-video summary must be on disk before the hard exit."""
        if not pre_video_written:
            write_artifacts(build_summary("aborted_before_video_finalization"))
        state("finalization_aborted")

    terminal_raw_libero_success = None
    with termination_as_interrupt():
        try:
            state("starting")
            with _simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            tasks = sim.list_tasks()
            selected = [task for task in tasks if task.get("task_id") == task_spec.task_id]
            if len(selected) != 1 or selected[0].get("prompt") != task_spec.prompt:
                raise ValueError("Official task does not match the selected task profile")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(task_spec.task_id, episode_index)
            collision = _collision_record(sim.status, speed_first=True)
            if obs.prompt != task_spec.prompt:
                raise ValueError("Reset observation has a different task")
            initial_observation = obs.to_dict()
            initial_sha256 = hashlib.sha256(json.dumps(initial_observation, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            write_json(output / "initial_observation.json", initial_observation)
            write_json(output / "initial_observation_check.json", {"sha256": initial_sha256,
                "expected_sha256": expected, "matched": initial_sha256 == expected if expected else None})
            if expected and initial_sha256 != expected:
                raise ValueError("Reset observation differs from the frozen historical comparison; no inference performed")
            policy.reset()
            recorder = recorder_factory(output / "episode.mp4", episode_id=(
                f"{'SCRIPTED SIM SMOKE' if smoke else 'CHATGPT SUBSCRIPTION'} / "
                f"{task_spec.suite}_t{task_spec.task_id:02d}_i{episode_index:02d} / AGENT SPEEDUP V1 / "
                + ("NO MODEL CALLS" if smoke else "COLLISIONS RECORDED")), offline_smoke=smoke)
            recorder.observe(obs, step=0)
            control_started = time.monotonic()
            with trace_path.open("w") as log:
                append = lambda row: _append(log, row)
                append({"event": "reset", "task_id": task_spec.task_id, "episode_index": episode_index,
                        "observation": obs.to_dict(), "sim_status": dict(sim.status), "mode": mode})
                while True:
                    collision = _collision_record(sim.status, speed_first=True)
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
                    record_decision_time(latency, details)
                    controller = details["controller"]
                    if controller != "agent_skill":
                        raise ValueError("agent_speedup_v1 decisions must use the agent_skill controller")
                    macro, primitive = details.get("macro"), details.get("primitive")
                    action = [float(value) for value in decision.action]
                    act_outcomes = details.get("outcomes")
                    if act_outcomes is None:
                        act_outcomes = [details["outcome"]] if details.get("outcome") is not None else []
                    for outcome in act_outcomes:
                        record = {"n": outcome.get("n"), "step": steps,
                                  "skill": outcome.get("skill", outcome.get("macro")),
                                  "status": outcome.get("status"), "reason": outcome.get("reason")}
                        outcomes.append(record)
                        append({"event": "macro_outcome", "step": steps, "outcome": outcome, "mode": mode})
                        note_outcome(record)
                    if details.get("refreshed"):
                        refreshes += 1
                        routings = details.get("routings") or [details["routing"]]
                        for index, routing in enumerate(routings[:-1]):
                            append({"event": "skill_decision", "step": steps, "index": index, "controller": controller,
                                    "routing": routing, "superseded_in_act": True,
                                    "source": "scripted_offline_smoke" if smoke else "model_prediction", "mode": mode})
                            note_decision(routing, True)
                        append({"event": "skill_decision", "step": steps, "controller": controller,
                                "routing": routings[-1], "macro": macro, "primitive": primitive,
                                "api_calls_in_act": details.get("api_calls_in_act"),
                                "api_latency_s": details.get("api_latency_s"),
                                "api_request_key": details.get("api_request_key"),
                                "api_request_log": details.get("api_request_log"),
                                "source": "scripted_offline_smoke" if smoke else "model_prediction", "mode": mode})
                        note_decision(routings[-1], False)
                    for index, result in enumerate(details.get("pointing") or []):
                        pointing_count += 1
                        append({"event": "pointing", "step": steps, "index": index, "result": result, "mode": mode})
                    append({"event": "action", "step": steps, "state": list(obs.state), "action": action,
                            "diagnostics": details, "decision_latency_s": latency, "mode": mode})
                    video_began = time.monotonic()
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    timings["video_render_s"] += time.monotonic() - video_began
                    pending_action = steps
                    state("executing")
                    before = obs
                    collision_was_detected = bool(collision and collision.get("detected"))
                    sim_began = time.monotonic()
                    obs, success = sim.step(decision.action)
                    timings["simulator_step_s"] += time.monotonic() - sim_began
                    executed_step, steps = steps, steps + 1
                    pending_action = None
                    collision = _collision_record(sim.status, speed_first=True)
                    if success != sim.status["success"]:
                        raise ValueError("Step result must match the configured simulator success criterion")
                    success_seen = success_seen or sim.status["libero_success"]
                    close_actions += int(action[6] > 0)
                    open_actions += int(action[6] < 0)
                    neutral_actions += int(action[6] == 0)
                    path_length += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    append({"event": "executed", "step": executed_step, "controller": controller,
                            "macro": macro, "primitive": primitive, "next_state": list(obs.state),
                            "observation": obs.to_dict(), "action": action, "success": success,
                            "sim_status": dict(sim.status), "mode": mode})
                    if collision["detected"] and not collision_was_detected:
                        append({"event": "recorded_collision", "step": executed_step, "collision": collision,
                                "collision_policy": "record_only", "libero_success": sim.status["libero_success"],
                                "affects_completion": False})
                    if first_close_event is None and action[6] > 0 and macro != "calibrate":
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
                            "macro": macro, "primitive": primitive, "skill_decisions": refreshes,
                            "close_actions": close_actions, "success_observed": success,
                            "api_attempts": client.budget_snapshot()["api_calls"]}
                    if success or steps % EXECUTED_CONSOLE_EVERY == 0:
                        print(json.dumps(line), flush=True)
                        console_pending = None
                    else:
                        console_pending = line
                    if success:
                        break
        except KeyboardInterrupt as exc:
            # Assignments only: I/O waits until the finalization signal deferral is installed.
            stop_source = "sigterm" if str(exc) == f"Received signal {int(signal.SIGTERM)}" else "sigint"
            status, caught, interrupted_stage = "user_stop", exc, stage
            error = {"type": "KeyboardInterrupt", "stage": stage, "source": stop_source,
                     "message": "User interrupted the agent_speedup_v1 episode"}
        except (APIError, MissingCredentials, BudgetExceeded, SchemaError) as exc:
            status, error, caught = "subscription_error", _safe_error(exc, stage), exc
        except Exception as exc:
            status, error, caught = "infrastructure_error", _safe_error(exc, stage), exc
        finally:
            if control_started is not None and control_stopped is None:
                control_stopped = time.monotonic()
            # R1: a second Ctrl+C / SIGTERM must not lose summary.json. Deferred until episodes.jsonl is written.
            signals = _FinalizationSignals(presses=int(isinstance(caught, KeyboardInterrupt)),
                                           on_abort=abort_finalization).install()
            try:
                if caught is not None:
                    trace_error(caught, interrupted_stage or stage)
                try:
                    terminal_raw_libero_success = sim.status.get("libero_success") if sim is not None else None
                except Exception:
                    terminal_raw_libero_success = None
                # In memory first, so even an abort during sim.close() leaves a summary with every outcome.
                guarded("finalization_trace_events", finalization_trace_events, [])
                if sim is not None:
                    try:
                        sim.close()
                    except BaseException as exc:  # recorded, never re-raised
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
                        # A video that fails to finalize never changes the episode outcome (success / user_stop /
                        # diagnostic); it is recorded separately.
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
    print(json.dumps({"event": "finished", **summary}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--task", choices=DISTANCE_TASK_KEYS, default="bowl_plate")
    parser.add_argument("--episode-index", type=int, choices=range(50), default=0,
                        help="LIBERO initial-state index (0..49); use the same index for paired arms")
    parser.add_argument("--expected-initial-observation-sha256",
                        help="Abort before inference if the RGB/state reset observation differs from the frozen reference")
    parser.add_argument("--target-description",
                        help="Optional user-provided target appearance from initial RGB")
    parser.add_argument("--local-retries", type=int, default=2,
                        help="Local grasp retries before a failure is reported to the model (0..5)")
    parser.add_argument("--external-axes", choices=EXTERNAL_AXES_MODES, default="measured", help=EXTERNAL_AXES_HELP)
    parser.add_argument("--no-video", action="store_true", help="Skip the annotated episode video")
    parser.add_argument("--smoke-script", choices=SMOKE_SCRIPTS,
                        help="--sim-smoke script: plumbing (reach/retreat) or grasp (bowl_plate init 0 rim pinch)")
    parser.add_argument("--smoke-steps", type=int,
                        help=f"Action cap for --sim-smoke (default {DEFAULT_SMOKE_STEPS})")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true",
                       help="Print the decision and pointing payloads; no credentials, simulator or files")
    modes.add_argument("--sim-smoke", action="store_true",
                       help="Real simulator with scripted offline responses; zero model calls, unscored")
    args = parser.parse_args(argv)
    if (args.smoke_steps is not None or args.smoke_script is not None) and not args.sim_smoke:
        parser.error("--smoke-steps/--smoke-script require --sim-smoke")
    smoke_script = args.smoke_script or "plumbing"
    if args.sim_smoke and smoke_script == "grasp" and (args.task != "bowl_plate" or args.episode_index != 0):
        parser.error("--smoke-script grasp is defined only for --task bowl_plate --episode-index 0")
    if args.sim_smoke and (args.store_dir is not None or args.profile is not None):
        parser.error("--sim-smoke never reads credentials; remove --store-dir/--profile")
    if not args.preflight and args.output is None:
        parser.error("--output is required for execution")
    config = configuration()
    try:
        sim_config = build_sim_config(args.task, sim_python=args.sim_python, libero_root=args.libero_root)
        target_description = normalize_target_description(args.target_description)
        _validate_local_retries(args.local_retries)
        _validate_expected_sha(args.expected_initial_observation_sha256)
        if args.sim_smoke:
            _validate_smoke_steps(args.smoke_steps if args.smoke_steps is not None else DEFAULT_SMOKE_STEPS)
        validate_configuration(config, sim_config, resolve_distance_task(args.task))
    except ValueError as exc:
        parser.error(str(exc))
    if args.preflight:
        try:
            result = preflight(config, sim_config, args.output, task_key=args.task, episode_index=args.episode_index,
                               target_description=target_description, local_retries=args.local_retries,
                               video=not args.no_video, external_axes=args.external_axes)
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    try:
        check_render_environment(warn=lambda *a, **k: print(*a, file=sys.stderr, **k))
    except ValueError as exc:
        parser.error(str(exc))
    if args.sim_smoke:
        token_provider = offline_token_provider
    else:
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
    result = execute(args.output, config=config, sim_config=sim_config, task_key=args.task,
        episode_index=args.episode_index, token_provider=token_provider,
        expected_initial_observation_sha256=args.expected_initial_observation_sha256,
        target_description=target_description, local_retries=args.local_retries, video=not args.no_video,
        smoke=args.sim_smoke, smoke_script=smoke_script, external_axes=args.external_axes,
        smoke_steps=(args.smoke_steps if args.smoke_steps is not None
                     else GRASP_SMOKE_STEPS if smoke_script == "grasp" else DEFAULT_SMOKE_STEPS))
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
