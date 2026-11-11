"""Isolated RGB foundation-vision adaptation of the existing mobile UC7.1 pilot.

The original simulator, mobile controller, raw success evaluator, record-only
contacts and uncapped loop remain authoritative. No previous policy transcript
or task/object truth enters the vision or executive request.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

from gpt_field.action import USER_CHECK_TRANSIT_ACTION_M
from gpt_field.astra import AstraConfig
from gpt_field.foundation_vision.bridge import file_sha256
from gpt_field.foundation_vision.live import FoundationVisionAgentClient, VISION_INSTRUCTIONS

from . import runner as common
from .policy import ROBOT_CONTEXT, RoboCasa365Client, RoboCasa365PlanClient, RoboCasa365Policy


VERSION = "rc365-foundation-vision-mobile-1"
METHOD = "foundation_vision_uc71_transit50_robocasa365_mobile"
PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class RoboCasaFoundationVisionClient(RoboCasa365Client, FoundationVisionAgentClient):
    """Mobile schema/context is applied after the foundation-vision payload."""

    prompt_version = VERSION

    def __init__(self, *args, **kwargs):
        self._deferred_vision_payload = None
        super().__init__(*args, **kwargs)
        self._instruction_text["agent"] = VISION_INSTRUCTIONS.replace(
            "gripper0_grip_site", "gripper0_right_grip_site") + "\n" + ROBOT_CONTEXT
        self._vision_stats.update(robot="PandaOmron", mobile_base=True, torso=True,
                                  mobile_geometry_check=False, adapter_version=VERSION)
        self._write_stats()

    def _append(self, name, record):
        # FV constructs its payload before the outer RoboCasa method adds the
        # mobile schema/context. Delay this one record until the actual wire
        # payload is final, rather than recording a misleading pre-adapter hash.
        if name == "payloads.jsonl":
            self._deferred_vision_payload = deepcopy(record)
        else:
            super()._append(name, record)

    def build_payload(self, request, grid=None, mode="plan"):
        if mode != "agent":
            return super().build_payload(request, grid, mode)
        self._deferred_vision_payload = None
        try:
            payload = super().build_payload(request, grid, mode)
            if self._deferred_vision_payload is None:
                raise RuntimeError("Missing foundation-vision payload audit record")
            record = self._deferred_vision_payload
            record["payload_sha256"] = hashlib.sha256(_json(payload).encode()).hexdigest()
            record["adapter_version"] = VERSION
            record["meaning"] = "final_mobile_and_foundation_vision_wire_payload_built; acceptance is in requests.jsonl"
            super()._append("payloads.jsonl", record)
            return payload
        finally:
            self._deferred_vision_payload = None


class StopSafeFoundationVisionPolicy(RoboCasa365Policy):
    """A stop arriving during inference cannot dispatch the resulting action."""

    def __init__(self, *args, stop_file, **kwargs):
        self.foundation_stop_file = Path(stop_file)
        self.withheld_decisions_after_stop = 0
        super().__init__(*args, **kwargs)

    def act(self, observation):
        if self.foundation_stop_file.exists():
            raise KeyboardInterrupt
        decision = super().act(observation)
        if self.foundation_stop_file.exists():
            self.withheld_decisions_after_stop += 1
            self._pending = None
            self._primitive = None
            self._need_agent = True
            self._record_feedback({"action": "operator_stop", "status": "interrupted_before_execution",
                                   "executed_steps": 0, "accepted_decision_dispatched": False})
            raise KeyboardInterrupt
        return decision

    def agent_stats(self):
        result = super().agent_stats()
        result["counts"]["withheld_decisions_after_stop"] = self.withheld_decisions_after_stop
        result["versions"]["foundation_vision_mobile"] = VERSION
        return result


def check_initial(observation, snapshot, reference_run):
    """Evaluator-side exact reset assertions, never old policy inputs/history."""
    reference = Path(reference_run)
    expected_sensors = json.loads((reference / "initial_sensors.json").read_text())
    expected_physics = json.loads((reference / "initial_physics.json").read_text())
    actual_sensors = common.sensor_record(observation)
    checks = {"physical_sha256": snapshot["physical_sha256"] == expected_physics["physical_sha256"],
              "model_xml_sha256": snapshot["model_xml_sha256"] == expected_physics["model_xml_sha256"],
              "complete_public_sensors": actual_sensors == expected_sensors,
              "official_prompt": observation.prompt == common.TASK_PROMPT}
    return {"reference_run": str(reference.resolve()), "checks": checks, "matched": all(checks.values()),
            "actual_sensor_sha256": common.canonical_sha256(actual_sensors),
            "expected_sensor_sha256": common.canonical_sha256(expected_sensors),
            "physical_sha256": snapshot["physical_sha256"], "model_xml_sha256": snapshot["model_xml_sha256"],
            "previous_policy_history_read": False, "reference_data_to_policy": False}


class _ReferenceCheckedSimulator:
    def __init__(self, simulator, reference_run, report_path):
        self.simulator, self.reference_run, self.report_path = simulator, reference_run, report_path

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    def reset(self):
        observation = self.simulator.reset()
        report = check_initial(observation, self.simulator.snapshot(include_xml=True), self.reference_run)
        common.write_json(self.report_path, report)
        if not report["matched"]:
            raise common.ReviewError("Fresh reset differs from the reference public sensors or physical/XML hashes")
        return observation


def configuration():
    return AstraConfig(model="gpt-6-astra", timeout_s=180,
                       max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)


def build_engine(args):
    from gpt_field.foundation_vision.backends import DA3DepthEstimator, Sam31Segmenter
    from gpt_field.foundation_vision.engine import EngineConfig, VisionEngine

    sam = Sam31Segmenter(checkpoint_path=args.sam_checkpoint, device=args.vision_device, local_files_only=True)
    da3 = DA3DepthEstimator(model_id=str(args.da3_model), device=args.vision_device, local_files_only=True)
    timings = {}
    for name, backend in (("sam", sam), ("da3", da3)):
        start = perf_counter()
        backend._load()
        timings[name + "_load_s"] = perf_counter() - start
    return VisionEngine(sam, da3, EngineConfig(max_age_s=args.vision_max_age_s)), {
        "load_times_s": timings, "models": {"sam": sam.provenance(), "da3": da3.provenance()}}


def settings(args, preload):
    identities = {name: {key: data.get(key) for key in
        ("backend", "model_id", "revision", "checkpoint_sha256", "precision")}
        for name, data in preload.get("models", {}).items()}
    return {"version": VERSION, "method": METHOD, "mode": "vision_conditioned_executive",
            "sam_checkpoint": str(args.sam_checkpoint), "da3_model": str(args.da3_model),
            "device": args.vision_device, "max_age_s": args.vision_max_age_s,
            "models": identities, "local_files_only": True,
            "target_source": "fresh_automatically_confirmed_plan_parts", "extra_queries": [],
            "initial_plan_vision_feedback": False, "sensor_depth_to_policy": False,
            "rgb_predicted_geometry_to_executive": True, "automatic_action_gate": False,
            "automatic_depth_routing": False, "obstacle_map_source": "astra_model_estimate",
            "mobile_base": True, "torso": True, "mobile_geometry_check": False,
            "full_robot_collision_check": False, "transit_action_m": .05,
            "reference_run": str(args.reference_run), "reference_use": "initial_reset_hashes_and_public_sensors_only",
            "human_reviewed_plan": False, "reasoning_effort": "high", "subscription_only": True,
            "episode_caps": None, "automatic_episode_retries": 0,
            "policy_class": "StopSafeFoundationVisionPolicy"}


@contextmanager
def installed_runner(vision_settings, reference_run, trial, client_holder):
    """Bind the new entry/config before both review and execution; restore globals."""
    inventory_before, writer_before, method_before = common.runtime_inventory, common.write_json, common.METHOD

    def inventory(config, sim_config):
        result = dict(inventory_before(config, sim_config))
        paths = [PROJECT / "run_foundation_vision.py", Path(__file__),
                 Path(reference_run) / "initial_sensors.json", Path(reference_run) / "initial_physics.json"]
        for path in paths:
            result[str(path.resolve())] = file_sha256(path)
        result["@foundation_vision"] = common.canonical_sha256(vision_settings)
        return result

    def write_json(path, value):
        path = Path(path)
        if path.parent == Path(trial) and path.name in {"manifest.json", "summary.json"}:
            value = {**value, "foundation_vision": deepcopy(vision_settings)}
            if path.name == "manifest.json":
                value["policy_inputs"] += "; SAM3.1 masks and DA3 RGB-predicted partial surface geometry"
                value["runtime_client"] = "RoboCasaFoundationVisionClient"
                value["runtime_policy"] = "StopSafeFoundationVisionPolicy"
            elif client_holder:
                value["foundation_vision_stats"] = client_holder["client"].vision_stats()
        return writer_before(path, value)

    common.runtime_inventory, common.write_json, common.METHOD = inventory, write_json, METHOD
    try:
        yield inventory
    finally:
        common.runtime_inventory, common.write_json, common.METHOD = inventory_before, writer_before, method_before


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Freshness age must be finite and positive")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("preflight", "start"))
    result.add_argument("--run-root", type=Path)
    result.add_argument("--sam-checkpoint", type=Path, required=True)
    result.add_argument("--da3-model", type=Path, required=True)
    result.add_argument("--vision-device", default="cuda:0")
    result.add_argument("--vision-max-age-s", type=_positive, default=5.)
    result.add_argument("--reference-run", type=Path, default=PROJECT / "runs/restock_pantry_001")
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--store-dir", type=Path)
    result.add_argument("--profile")
    result.add_argument("--no-video", action="store_true")
    result.add_argument("--test-approved", action="store_true")
    return result


def validate_args(args):
    if not math.isclose(USER_CHECK_TRANSIT_ACTION_M, .05, abs_tol=1e-12):
        raise ValueError("Use run_foundation_vision.py: transit50 must be configured before importing gpt_field")
    args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    args.da3_model = args.da3_model.expanduser().resolve()
    args.reference_run = args.reference_run.expanduser().resolve()
    if not args.sam_checkpoint.is_file():
        raise ValueError("Local official SAM 3.1 checkpoint is required")
    if not all((args.da3_model / name).is_file() for name in ("config.json", "model.safetensors")):
        raise ValueError("Local DA3 model directory requires config.json and model.safetensors")
    if not all((args.reference_run / name).is_file() for name in ("initial_sensors.json", "initial_physics.json")):
        raise ValueError("Reference requires only initial_sensors.json and initial_physics.json")
    if args.run_root is not None:
        args.run_root = args.run_root.expanduser().resolve()


def preflight(args):
    from gpt_field.foundation_vision.backends import DA3DepthEstimator, Sam31Segmenter
    config, sim_config = configuration(), common.RoboCasaConfig(seed=args.seed)
    vision_settings = settings(args, {})
    with installed_runner(vision_settings, args.reference_run, Path("/__not_a_trial__"), {}):
        inventory = common.runtime_inventory(config, sim_config)
    sam = Sam31Segmenter(checkpoint_path=args.sam_checkpoint, device=args.vision_device, local_files_only=True)
    da3 = DA3DepthEstimator(model_id=str(args.da3_model), device=args.vision_device, local_files_only=True)
    return {"method": METHOD, "inspection_only": True, "models_loaded": False, "model_calls": 0,
            "simulator_starts": 0, "credential_reads": 0, "foundation_vision": vision_settings,
            "api": asdict(config), "simulator": asdict(sim_config),
            "model_preflight": {"sam": sam.preflight(), "da3": da3.preflight()},
            "runtime_sha256": inventory, "requires_test_approved_for_start": True}


def credential_provider(args):
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    return SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token


def start(args, *, engine_factory=build_engine, auth_factory=credential_provider,
          sim_factory=None, plan_client_factory=RoboCasa365PlanClient,
          execution_client_factory=RoboCasaFoundationVisionClient, policy_factory=StopSafeFoundationVisionPolicy):
    if not args.test_approved:
        raise ValueError("start requires --test-approved for this explicitly authorized one-shot test")
    if args.run_root is None:
        raise ValueError("start requires --run-root")
    root = Path(args.run_root).resolve()
    for name in ("review", "trial", "launch_state.json", ".foundation_attempt_started.json"):
        if (root / name).exists():
            raise FileExistsError(f"Fresh one-shot paths required; {name} already exists")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".foundation_attempt_started.json").open("x", encoding="utf-8") as stream:
        stream.write(_json({"version": VERSION, "pid": os.getpid(), "test_approved": True,
                            "automatic_retry": False}) + "\n")
    review, trial = root / "review", root / "trial"
    config, sim_config = configuration(), common.RoboCasaConfig(seed=args.seed)
    holder, reset_count = {}, 0

    def state(stage, **extra):
        value = {"stage": stage, "method": METHOD, "pid": os.getpid(),
                 "at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(root),
                 "review_dir": str(review), "trial": str(trial), "automatic_retry": False, **extra}
        common.write_json(root / "launch_state.json", value)
        print(_json(value), flush=True)

    def checked_sim(simulator_config):
        nonlocal reset_count
        reset_count += 1
        simulator = (sim_factory or common.RoboCasaSimProcess)(simulator_config)
        return _ReferenceCheckedSimulator(simulator, args.reference_run,
                                          root / f"initial_match_{reset_count:02d}.json")

    try:
        with common.reviewed.base.termination_as_interrupt():
            state("preloading_local_foundation_models")
            engine, preload = engine_factory(args)
            common.write_json(root / "vision_preload.json", preload)
            vision_settings = settings(args, preload)
            state("models_ready_before_auth_and_reset")
            token_provider = auth_factory(args)
            with installed_runner(vision_settings, args.reference_run, trial, holder):
                state("preparing_fresh_initial_plan", human_reviewed_plan=False)
                draft = common.prepare(review, config=config, sim_config=sim_config,
                    token_provider=token_provider, sim_factory=checked_sim,
                    client_factory=plan_client_factory, auto_approve=True)
                if (root / common.reviewed.base.OPERATOR_STOP_FILE).exists() or (trial / common.reviewed.base.OPERATOR_STOP_FILE).exists():
                    raise KeyboardInterrupt
                store = common.ReviewStore(review)
                approval = store.approve(expected_revision=draft["revision"],
                                         reviewed_steps=list(range(len(draft["plan"]["steps"]))))
                common.write_json(root / "automatic_plan_confirmation.json", {
                    "approval_mode": "user_authorized_automatic", "manual_review": False,
                    "human_reviewed_plan": False, "review_id": approval["id"],
                    "plan_sha256": approval["approved"]["plan_sha256"], "single_use": True})

                def client_factory(*positional, **kwargs):
                    client = execution_client_factory(*positional, vision_engine=engine,
                        vision_output_dir=trial / "foundation_vision", **kwargs)
                    holder["client"] = client
                    return client

                def safe_policy(*positional, **kwargs):
                    return policy_factory(*positional,
                        stop_file=trial / common.reviewed.base.OPERATOR_STOP_FILE, **kwargs)

                state("executing_full_task", human_reviewed_plan=False, foundation_vision=vision_settings)
                summary = common.execute(trial, review_dir=review, config=config, sim_config=sim_config,
                    token_provider=token_provider, sim_factory=checked_sim, client_factory=client_factory,
                    policy_factory=safe_policy, video=not args.no_video)
            state("terminal", status=summary["status"], steps=summary["steps"],
                  task_success=summary["task_success"], raw_robocasa_success=summary["robocasa_success"])
            return summary
    except KeyboardInterrupt:
        state("terminal", status="user_stop", task_success=False)
        return {"status": "user_stop", "task_success": False, "robocasa_success": False}
    except Exception as exc:
        state("startup_error", status="error", error_type=type(exc).__name__,
              error=common.reviewed.base.redact_secrets(str(exc)))
        raise


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.command == "start" and (not args.test_approved or args.run_root is None):
        command_parser.error("start requires --test-approved and --run-root")
    validate_args(args)
    if args.command == "preflight":
        print(_json(preflight(args)))
        return 0
    result = start(args)
    print(_json(result), flush=True)
    return 0 if result["task_success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
