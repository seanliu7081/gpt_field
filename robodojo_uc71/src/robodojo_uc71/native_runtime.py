"""Restricted native worker bootstrap. Importing this module launches nothing.

run_native_worker requires a separate nobody process, no_new_privs, root-owned
read-only spec/manifest/human approval, a precreated empty worker directory, exact
pinned source/assets, and an evidence-backed injected runtime bundle. Diagnostic
asset assessment alone grants no full-test authorization. Published asset defects
remain explicit and require acceptance in the human-approved manifest.

The parent owns the socket/process/GPU watchdog, private Git safe.directory list
for exactly the four pinned repositories, runtime environment, approval creation,
consumption after actual READY, and post-reap reconciliation. This worker imports
no model client, makes no inference call, and never updates Kit while socket-idle.
Only NativeWorkerSession's verified ticket may reach a guarded native action.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import traceback

from .approval import canonical_sha256, file_sha256, verify_manifest
from .ipc import JsonLineChannel
from .native_config import (ResetOnlyClient, add_native_import_paths, build_native_preview_config,
                            install_preview_adapters, validate_native_preview_config)
from .native_episode import validate_joint_action
from .native_episode_runner import execution_configuration
from .native_measurements import capture_camera_inventory, capture_paused_physics_state
from .native_robot_inventory import capture_robot_inventory
from .native_worker_protocol import NativeWorkerSession, RuntimeReadiness, WorkerCapture
from .reset_binding import verify_fresh_reset
from .schema import CAMERAS, validate_calibration
from .worker_isolation import WORKER_GID, WORKER_UID, enforce_child_boundary

BASE = Path(__file__).resolve().parents[2]
SCRIPTS = BASE / "scripts"
TASK_INSTRUCTION = "Pick up two slices of bread, place them into the toaster, and press the lever down."
CREDENTIAL_STORE = Path("/workspace/gpt_field/.private/chatgpt_plan")


class NativeRuntimeError(RuntimeError):
    pass


def private_runtime_paths(runtime_dir: Path):
    """Mirror reviewed Kit107.3 token overrides, rooted in this worker only."""
    root = runtime_dir/"bootstrap/kit-portable"
    tokens = {}
    for kind in ("cache","data","logs"):
        tokens[kind] = str(root/kind/"kit")
        tokens["omni_"+kind] = tokens["omni_global_"+kind] = str(root/kind)
    return {"tmpdir":str(runtime_dir/"tmp"),"portable_root":str(root),"tokens":tokens,
        "isaaclab_log_directory":str(runtime_dir/"tmp/isaaclab/logs")}


def prepare_private_runtime_paths(runtime_dir: Path):
    expected = private_runtime_paths(runtime_dir); tmp = Path(expected["tmpdir"]); info = tmp.lstat()
    if (os.geteuid() != WORKER_UID or tmp.resolve(strict=True) != tmp or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != WORKER_UID or info.st_gid != WORKER_GID or stat.S_IMODE(info.st_mode) != 0o700
            or os.environ.get("TMPDIR") != str(tmp) or Path(tempfile.gettempdir()) != tmp):
        raise NativeRuntimeError("worker_private_tmpdir_required")
    for name in sorted(set(expected["tokens"].values())):
        path = Path(name); path.mkdir(parents=True,mode=0o700,exist_ok=True)
        if path.resolve(strict=True) != path or path.stat().st_uid != WORKER_UID:
            raise NativeRuntimeError("worker_private_kit_path_required")
    with tempfile.TemporaryFile(dir=tmp) as stream:
        stream.write(b"private-temp-path-check"); stream.flush()
    return {"schema":"robodojo-private-runtime-path-preparation-1","pid":os.getpid(),
        "uid":os.geteuid(),**expected,"temporary_file_write_passed":True,"kit_runtime_resolution_checked":False}


def private_kit_arguments(runtime_dir: Path):
    expected = private_runtime_paths(runtime_dir)
    # Kit107.3 requires a separate portable-root argument; tokens are checked live.
    return ["--portable","--portable-root",expected["portable_root"],
        *(f"--/app/tokens/{name}={value}" for name,value in expected["tokens"].items())]


def verify_effective_runtime_paths(runtime_dir: Path, resolve):
    expected = private_runtime_paths(runtime_dir)
    actual = {name:resolve("${"+name+"}") for name in expected["tokens"]}
    if actual != expected["tokens"] or str(Path(tempfile.gettempdir())) != expected["tmpdir"]:
        raise NativeRuntimeError("effective_runtime_paths_escaped_private_worker")
    return {"schema":"robodojo-private-runtime-paths-1","pid":os.getpid(),"uid":os.geteuid(),
        "monotonic_s":time.monotonic(),**expected,"resolved_tokens":actual,"verified_before_native_reset":True}


def verify_native_logging_paths(runtime_dir: Path, handlers):
    import logging
    expected = Path(private_runtime_paths(runtime_dir)["isaaclab_log_directory"])
    paths = [Path(handler.baseFilename) for handler in handlers if isinstance(handler,logging.FileHandler)]
    if not paths or any(path.parent != expected or path.resolve(strict=True) != path or not path.is_file()
            or path.stat().st_uid != WORKER_UID or not path.name.startswith("isaaclab_") for path in paths):
        raise NativeRuntimeError("native_file_logging_escaped_private_worker")
    return {"schema":"robodojo-private-native-logging-1","pid":os.getpid(),"uid":os.geteuid(),
        "monotonic_s":time.monotonic(),"directory":str(expected),"file_paths":[str(p) for p in paths],
        "verified_after_native_reset":True}


def _strict_pairs(rows):
    result = {}
    for key, value in rows:
        if key in result: raise NativeRuntimeError("duplicate_record_key")
        result[key] = value
    return result


def root_readonly_record(path: Path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_uid != 0 or info.st_gid != WORKER_GID
            or stat.S_IMODE(info.st_mode) not in (0o640, 0o440) or info.st_nlink != 1 or info.st_size > 16*1024*1024):
        raise NativeRuntimeError("root_owned_readonly_record_required")
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_strict_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite record")))
    if not isinstance(value, dict): raise NativeRuntimeError("record_object_required")
    return value


@dataclass(frozen=True)
class WorkerRuntimeSpec:
    manifest_path: Path
    runtime_bundle_path: Path
    native_root: Path
    source_manifest_path: Path
    asset_manifest_path: Path
    asset_assessment_path: Path
    runtime_dir: Path
    parent_log: Path
    gpu_binding: dict
    boot_nonce: str

    @classmethod
    def load(cls, path):
        value = root_readonly_record(Path(path))
        expected = {field.name for field in fields(cls)} | {"schema"}
        if set(value) != expected or value["schema"] != "robodojo-native-worker-spec-1":
            raise NativeRuntimeError("exact_worker_spec_required")
        value.pop("schema")
        for name in expected - {"schema", "gpu_binding", "boot_nonce"}:
            path = Path(value[name])
            if not path.is_absolute() or path.resolve(strict=True) != path or any(c.isspace() for c in str(path)):
                raise NativeRuntimeError("canonical_existing_worker_paths_required")
            value[name] = path
        binding = value["gpu_binding"]
        if (not isinstance(binding, dict) or set(binding) != {"gpu_uuid","pci_bus_id","kit_index","cuda_ordinal"}
                or type(binding["kit_index"]) is not int or binding["kit_index"] != 0
                or type(binding["cuda_ordinal"]) is not int or binding["cuda_ordinal"] != 0
                or not isinstance(binding["gpu_uuid"], str) or not binding["gpu_uuid"].startswith("GPU-")
                or not isinstance(binding["pci_bus_id"], str)
                or not isinstance(value["boot_nonce"], str) or re.fullmatch(r"[A-Za-z0-9_-]{16,128}",value["boot_nonce"]) is None):
            raise NativeRuntimeError("single_gpu_and_fresh_boot_nonce_required")
        return cls(**value)


def runtime_program_files():
    modules = ("native_runtime.py", "native_runtime_bundle.py", "native_config.py", "native_robot_inventory.py",
        "native_measurements.py", "native_worker_protocol.py", "native_episode.py", "native_episode_runner.py",
        "native_worker_client.py", "fragment_executor.py", "controller.py", "schema.py", "lifecycle.py",
        "approval.py", "ipc.py", "reset_binding.py", "worker_isolation.py")
    return [*(Path(__file__).with_name(name) for name in modules),
        *(SCRIPTS/name for name in ("run_native_worker.py","new_host_camera.py","prepare_native_preview.py","provision_native_common.py")),
        BASE/"configs/source_pins.json"]


def published_asset_policy(preparation: Mapping) -> dict:
    assessment = preparation.get("asset_diagnostic_assessment")
    if (not isinstance(assessment, dict) or not assessment.get("known_missing_reference")
            or preparation.get("asset_fidelity_passed") is not False
            or preparation.get("dependency_closure_complete") is not False):
        raise NativeRuntimeError("documented_published_asset_defect_required")
    return {"mode": "published_assets_with_documented_defect", "accept_published_asset_defects": True,
        "asset_manifest_sha256": preparation["asset_manifest_sha256"],
        "asset_assessment_sha256": assessment["sha256"],
        "known_missing_reference": assessment["known_missing_reference"],
        "authored_assets_modified": False, "asset_fidelity_passed": False, "dependency_closure_complete": False}


def verify_human_preparation_approval(manifest, preparation):
    verify_manifest(manifest)
    attempt = Path(manifest["attempt_dir"])
    approval = root_readonly_record(attempt/"approval.json")
    expected = {"schema":"robodojo-uc71-human-approval-1", "approval_mode":"manual_chat_review",
        "manual_browser_review":False, "manifest_sha256":manifest["manifest_sha256"],
        "preview_sha256":manifest["preview_sha256"], "attempt_dir":str(attempt), "attempts":1}
    if (any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k,v in expected.items())
            or not isinstance(approval.get("user_message"), str) or not approval["user_message"].strip()
            or manifest["configuration"].get("asset_policy") != published_asset_policy(preparation)
            or manifest["configuration"].get("task_prompt") != TASK_INSTRUCTION):
        raise NativeRuntimeError("human_approved_task_and_published_asset_policy_required")
    return {"path":str(attempt/"approval.json"),"sha256":file_sha256(attempt/"approval.json"),
            "manifest_sha256":manifest["manifest_sha256"], "human_approval_verified":True,
            "asset_fidelity_passed":False,"dependency_closure_complete":False}


class NativeExecutionGuard:
    """One setup reset; afterward only a verified session ticket can dispatch."""
    def __init__(self, env):
        self.env, self.session = env, None
        self.reset_used = self._reset_permit = self._reset_active = self._inside = False
        self._armed = None
        self.sim_step_calls = 0
        reset, take, step = env.reset, env.take_action, env.sim_step
        def forbidden(*args, **kwargs): raise NativeRuntimeError("native_episode_loop_or_restart_forbidden")
        for name in ("run_eval","eval_one_episode","eval_one_episode_batch"):
            if hasattr(env,name): setattr(env,name,forbidden)
        def guarded_reset(*args, **kwargs):
            if not self._reset_permit or self.reset_used: return forbidden()
            self.reset_used, self._reset_permit, self._reset_active = True, False, True
            try: return reset(*args, **kwargs)
            finally: self._reset_active = False
        def guarded_take(action):
            clean = validate_joint_action(action)
            if (self._inside or self.session is None or self.session.state != "RUNNING" or self._armed is None
                    or self.session.lifecycle.snapshot()["in_flight"] != self._armed[0]
                    or env.take_action_cnt != [self._armed[0]-1] or clean != self._armed[1]):
                raise NativeRuntimeError("only_verified_session_ticket_may_dispatch")
            self._armed = None; self._inside = True
            try: return take(clean)
            finally: self._inside = False
        def guarded_sim_step(*args, **kwargs):
            if not (self._reset_active or self._inside): raise NativeRuntimeError("unrequested_native_physics_step")
            self.sim_step_calls += 1
            return step(*args, **kwargs)
        env.reset, env.take_action, env.sim_step = guarded_reset, guarded_take, guarded_sim_step
        for name in ("take_action_batch","step"):
            native = getattr(env,name)
            def nested(*args, _native=native, **kwargs):
                if not self._inside: raise NativeRuntimeError("native_control_outside_verified_dispatch")
                return _native(*args, **kwargs)
            setattr(env,name,nested)

    def reset_once(self):
        if self.reset_used: raise NativeRuntimeError("native_reset_is_single_use")
        self._reset_permit = True
        return self.env.reset(seed=[0])

    def bind(self, session: NativeWorkerSession):
        if self.session is not None or not self.reset_used: raise NativeRuntimeError("guard_binding_is_single_use_after_reset")
        self.session = session
        original = session._authorize_action
        def authorize(ticket, action):
            if original(ticket, action) is not True: return False
            if self._armed is not None: raise NativeRuntimeError("unused_native_action_authorization")
            self._armed = (ticket, deepcopy(action))
            return True
        session._authorize_action = authorize


class RuntimeCallbacks:
    def __init__(self, *, env, bundle, manifest, preparation, approval, output, guard, monitor, config, save_rgb, write_json):
        self.env, self.bundle, self.manifest = env, bundle, manifest
        self.preparation, self.approval, self.output = preparation, approval, output
        self.guard, self.monitor, self.config = guard, monitor, config
        self._save_rgb, self._write = save_rgb, write_json
        self._starts = 0

    def _clock(self):
        return (float(self.env.sim.sim.current_time), self.env.sim._sim_step_counter,
                tuple(self.env.take_action_cnt), self.guard.sim_step_calls)

    def _snapshot(self, directory):
        before = self._clock()
        self.env.obs_manager.render_for_capture()
        rows = self._save_rgb(self.env.obs_manager.get_obs()[0], directory)
        images = {name: directory/next(row["path"] for row in rows if row["camera"] == "cam_"+name) for name in CAMERAS}
        robot, camera = capture_robot_inventory(self.env), capture_camera_inventory(self.env)
        measurement = self.bundle.measure(self.env)
        paused = capture_paused_physics_state(self.env, self.guard.sim_step_calls)
        paused.update(native_physics_counter=self.env.sim._sim_step_counter, native_action_counter=list(self.env.take_action_cnt))
        self._write(directory/"robot_inventory.json", robot)
        self._write(directory/"camera_inventory.json", camera)
        self._write(directory/"native_paused_state.json", paused)
        if before != self._clock(): raise NativeRuntimeError("runtime_capture_advanced_physics")
        return images, robot, camera, measurement

    def verify(self, phase):
        before = self._clock()
        if (self.bundle.verify() is not True or self.monitor.is_fatal() or self.monitor.get_broken_envs()
                or type(self.env.model_client) is not ResetOnlyClient or self.env.model_client.reset_calls != 1
                or self.env.robot_manager.planner or self.env.robot_manager.ik_solver):
            raise NativeRuntimeError("runtime_bundle_or_native_monitor_failed")
        validate_native_preview_config(self.config)
        root_readonly_record(Path(self.approval["path"]))
        current_manifest = root_readonly_record(Path(self.preparation["manifest_path"]))
        if current_manifest != self.manifest:
            raise NativeRuntimeError("approved_manifest_changed")
        consumed_path = Path(self.manifest["attempt_dir"])/"consumed.json"
        if phase in ("begin","dispatch") or consumed_path.exists():
            root_readonly_record(consumed_path)
        if file_sha256(Path(self.approval["path"])) != self.approval["sha256"]:
            raise NativeRuntimeError("human_approval_changed")
        if self.env.gen_instruction(0) != [TASK_INSTRUCTION]: raise NativeRuntimeError("native_task_instruction_changed")
        sensor = start = None
        if phase in ("ready","begin"):
            if self.env.take_action_cnt != [0]: raise NativeRuntimeError("fresh_zero_action_reset_required")
            self._starts += 1
            directory = self.output/f"fresh_reset_{self._starts:02d}_{phase}"
            directory.mkdir(mode=0o700, exist_ok=False)
            images, robot, camera, _ = self._snapshot(directory)
            comparison = verify_fresh_reset(self.bundle.reset_reference, robot, camera, images)
            self._write(directory/"reset_comparison.json", comparison)
            sensor, start = self.bundle.sensor_signature(camera), comparison["start_state_signature"]
            if not isinstance(sensor, Mapping): raise NativeRuntimeError("verified_sensor_mapping_required")
        if before != self._clock(): raise NativeRuntimeError("runtime_readiness_advanced_physics")
        return RuntimeReadiness(True,"validated",True,True,sensor,start)

    def measure(self):
        return self.bundle.measure(self.env)

    def capture(self, directory):
        images, _, camera, measured = self._snapshot(directory)
        return WorkerCapture(images, measured, validate_calibration(self.bundle.public_calibration(camera)))


def run_native_worker(spec: WorkerRuntimeSpec, *, ipc_fd: int, parent_pid: int, bundle_loader) -> int:
    """Launch exactly one human-approved worker, with no policy/model in-process."""
    boundary = enforce_child_boundary(credential_store=CREDENTIAL_STORE, parent_pid=parent_pid)
    if not callable(bundle_loader) or "CUDA_VISIBLE_DEVICES" in os.environ or os.environ.get("GIT_OPTIONAL_LOCKS") != "0":
        raise NativeRuntimeError("verified_factory_unmasked_gpu_and_readonly_git_required")
    manifest = root_readonly_record(spec.manifest_path)
    verify_manifest(manifest)
    attempt = Path(manifest["attempt_dir"])
    info = attempt.lstat()
    if (attempt.resolve(strict=True) != attempt or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0 or info.st_gid != WORKER_GID or stat.S_IMODE(info.st_mode) != 0o750):
        raise NativeRuntimeError("root_owned_readonly_attempt_required")
    runtime = spec.runtime_dir.lstat()
    if (spec.runtime_dir != attempt/"runtime" or not stat.S_ISDIR(runtime.st_mode)
            or runtime.st_uid != WORKER_UID or runtime.st_gid != WORKER_GID
            or stat.S_IMODE(runtime.st_mode) != 0o700): raise NativeRuntimeError("private_worker_runtime_required")
    required = [*runtime_program_files(), spec.runtime_bundle_path, spec.source_manifest_path,
                spec.asset_manifest_path, spec.asset_assessment_path]
    bound = {entry["path"]:entry["sha256"] for entry in manifest["files"].values()}
    if any(bound.get(str(path)) != file_sha256(path) for path in required):
        raise NativeRuntimeError("runtime_program_and_input_closure_not_approved")
    sys.path.insert(0,str(SCRIPTS))
    from new_host_camera import query_cuda_identity, renderer_table_identity, write_durable_json
    from prepare_native_preview import save_rgb_images, verify_manifests
    from provision_native_common import pins
    preparation = verify_manifests(spec.native_root, spec.source_manifest_path, spec.asset_manifest_path,
                                   pins(), spec.asset_assessment_path)
    approval = verify_human_preparation_approval(manifest, preparation)
    if preparation["camera_gate"]["gpu_uuid"] != spec.gpu_binding["gpu_uuid"]:
        raise NativeRuntimeError("approved_gpu_differs_from_verified_preparation")
    bundle = bundle_loader(spec.runtime_bundle_path,spec.native_root)
    if bundle.verify() is not True: raise NativeRuntimeError("runtime_bundle_not_verified")
    expected = execution_configuration(controller_config=bundle.controller_config, monitoring=bundle.monitoring_settings,
        limits=bundle.execution_limits, initial_gripper_commands={"left":"open","right":"open"},
        controller_capabilities=bundle.controller_capabilities)
    if manifest["configuration"].get("execution") != expected:
        raise NativeRuntimeError("approved_execution_differs_from_measured_bundle")
    output = spec.runtime_dir/"bootstrap"
    output.mkdir(mode=0o700, exist_ok=False)
    private_paths = prepare_private_runtime_paths(spec.runtime_dir)
    write_durable_json(output/"private_path_preparation.json",private_paths)
    os.chdir(output)
    run_id = "standard-" + spec.boot_nonce
    os.environ["ROBODOJO_RUN_ID"] = run_id
    add_native_import_paths(spec.native_root)
    config = build_native_preview_config(spec.native_root,run_id)
    from omegaconf import OmegaConf
    write_durable_json(output/"native_configuration.json",OmegaConf.to_container(config,resolve=True))
    preparation.update(schema="robodojo-native-worker-preparation-1",manifest_sha256=manifest["manifest_sha256"],
        manifest_path=str(spec.manifest_path),
        runtime_bundle_sha256=file_sha256(spec.runtime_bundle_path),human_approval=approval,
        binding=spec.gpu_binding,worker_identity={"pid":os.getpid(),"boot_nonce":spec.boot_nonce},
        asset_policy=published_asset_policy(preparation),source_file_sha256={str(p):file_sha256(p) for p in required})
    write_durable_json(output/"preparation.json",preparation)
    write_durable_json(output/"worker_isolation.json",boundary)
    app = env = monitor = session = channel = guard = None
    failure = None; cleanup_errors = []
    try:
        write_durable_json(output/"cuda_identity.json",query_cuda_identity(spec.gpu_binding))
        from src.eval_client.physx_warning_monitor import get_monitor
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args, zero_delay_active
        from isaaclab.app import AppLauncher
        monitor = get_monitor(); monitor.start(enabled=True)
        kit_args = [*private_kit_arguments(spec.runtime_dir),f"--/log/file={output/'kit.log'}",
            "--/app/settings/persistent=false","--/app/telemetry/enable=false","--/telemetry/enableAnonymousData=false",
            "--/telemetry/enableAnonymousAppName=false","--/rtx/multiGpu/enabled=false","--/rtx/multiGpu/maxGpuCount=1"]
        launcher = argparse.Namespace(headless=True,enable_cameras=True,livestream=0,device="cpu",multi_gpu=False,
            width=640,height=480,fast_shutdown=True,enable_crashreporter=False,kit_args=" ".join(kit_args))
        add_zero_delay_kit_args(launcher)
        write_durable_json(output/"app_configuration.json",vars(launcher))
        app = AppLauncher(launcher).app
        import carb.tokens
        write_durable_json(output/"runtime_paths.json",
            verify_effective_runtime_paths(spec.runtime_dir,carb.tokens.get_tokens_interface().resolve))
        import carb.settings
        settings = carb.settings.get_settings()
        observed = {key:settings.get(key) for key in ("/renderer/activeGpu","/physics/cudaDevice","/renderer/multiGpu/enabled")}
        write_durable_json(output/"observed_gpu_settings.json",observed)
        if observed != {"/renderer/activeGpu":0,"/physics/cudaDevice":0,"/renderer/multiGpu/enabled":False}:
            raise NativeRuntimeError("native_worker_gpu_settings_differ")
        renderer = renderer_table_identity(spec.parent_log.read_text(errors="replace"),spec.gpu_binding)
        write_durable_json(output/"renderer_identity.json",renderer)
        if renderer["uuid_prefix_and_pci_bus_match"] is not True or not zero_delay_active():
            raise NativeRuntimeError("native_renderer_or_camera_synchronization_failed")
        adaptations = install_preview_adapters()
        from src.eval_client.eval_env import create_eval_env
        env = create_eval_env(config,app,resume_state=None)
        if type(env.model_client) is not ResetOnlyClient or env.robot_manager.planner or env.robot_manager.ik_solver:
            raise NativeRuntimeError("native_model_transport_and_planners_must_be_disabled")
        guard = NativeExecutionGuard(env); guard.reset_once()
        import logging
        write_durable_json(output/"native_logging_paths.json",
            verify_native_logging_paths(spec.runtime_dir,logging.getLogger().handlers))
        if (env.device != "cpu" or env.use_fabric is not False or env.num_envs != 1 or list(env.env_seeds) != [0]
                or env.take_action_cnt != [0] or env.model_client.reset_calls != 1 or env.step_lim != 1400):
            raise NativeRuntimeError("native_standard_layout0_reset_failed")
        selected = Path(env.seed_manager.seed_info[0]["scene_layout"]).resolve(strict=True)
        if str(selected) != preparation["layout_path"] or file_sha256(selected) != preparation["layout_sha256"]:
            raise NativeRuntimeError("native_layout0_changed")
        instructions = env.gen_instruction(0)
        if instructions != [TASK_INSTRUCTION]: raise NativeRuntimeError("native_instruction_differs")
        write_durable_json(output/"task_instruction.json",{"schema":"robodojo-native-task-instruction-1",
            "task_name":"make_toast","variant":"Standard","layout":0,"instructions":instructions,
            "source":"native env.gen_instruction(0) after reset"})
        env.run_reward(); env.get_score()  # Register native predicates/score; do not evaluate them.
        env._check_physx_broken_envs()
        if monitor.is_fatal() or monitor.get_broken_envs(): raise NativeRuntimeError("native_physx_monitor_failed")
        registration_id = manifest["manifest_sha256"] + ":make_toast-registered-native-predicates"
        write_durable_json(output/"registration.json",{"registration_evidence_id":registration_id,
            "native_action_count":0,"reward_evaluations_added_by_bootstrap":0})
        modules = {}
        for name in ("numpy","torch","omegaconf","gymnasium","shapely","transforms3d","h5py",
                     "isaaclab","isaaclab_tasks","isaacsim","websockets","hid","setuptools","pkg_resources","_distutils_hack"):
            module = sys.modules.get(name)
            if module is not None:
                modules[name] = {"file":str(getattr(module,"__file__",None)),"version":str(getattr(module,"__version__",None))}
        write_durable_json(output/"loaded_module_provenance.json",modules)
        callbacks = RuntimeCallbacks(env=env,bundle=bundle,manifest=manifest,preparation=preparation,
            approval=approval,output=output,guard=guard,monitor=monitor,config=config,
            save_rgb=save_rgb_images,write_json=write_durable_json)
        channel = JsonLineChannel.from_fd(ipc_fd,role="worker",timeout_s=30.)
        session = NativeWorkerSession(channel,env=env,manifest=manifest,
            worker_identity={"pid":os.getpid(),"boot_nonce":spec.boot_nonce},verify_runtime_readiness=callbacks.verify,
            measure_robot=callbacks.measure,capture_rgb=callbacks.capture,expected_layout_path=selected,
            expected_layout_sha256=preparation["layout_sha256"],registration_evidence_id=registration_id,
            initial_gripper_commands={"left":"open","right":"open"},precreated_worker_output=True)
        guard.bind(session)
        write_durable_json(output/"runtime_initialized.json",{"adaptations":adaptations,
            "reset_calls":1,"model_calls":0,"native_policy_actions":0,"waiting_for_actual_ready_request":True})
        while session.serve_one():
            pass  # Socket wait only: never app.update/render/step while idle.
    except BaseException as error:
        failure = {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        write_durable_json(output/"worker_failure.json",failure)
    finally:
        if channel is not None: channel.close()
        try:
            graphics = sorted({line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
                if len(line.split(maxsplit=5)) == 6 and re.search(r'/(?:libcuda|libnvidia|libGLX|libEGL|libGLU|libvulkan)',line.split(maxsplit=5)[5])})
            write_durable_json(output/"loaded_graphics_library_paths.json",{"pid":os.getpid(),"paths":graphics,"source":"/proc/self/maps"})
        except BaseException as error:
            cleanup_errors.append({"phase":"graphics_provenance","type":type(error).__name__})
        for phase, operation in (("native_env_close",env.close if env is not None else None),
                                 ("physx_monitor_shutdown",monitor.shutdown if monitor is not None else None)):
            if operation is not None:
                try: operation()
                except BaseException as error: cleanup_errors.append({"phase":phase,"type":type(error).__name__})
        healthy = monitor is not None and not monitor.is_fatal() and not monitor.get_broken_envs()
        terminal_path = attempt/"worker/terminal.json"
        ready = bool(failure is None and not cleanup_errors and healthy and terminal_path.is_file())
        evidence = {"schema":"robodojo-native-worker-preclose-1","ready_for_shutdown":ready,
            "no_pending_exception":failure is None and sys.exc_info()[0] is None,
            "native_env_close_returned":env is not None and not any(x["phase"] == "native_env_close" for x in cleanup_errors),
            "physx_monitor_healthy":healthy,"cleanup_errors":cleanup_errors,"child_pid":os.getpid(),
            "binding":spec.gpu_binding,"shutdown_mode":"stock-fast","full_teardown_returned":False,
            "full_plugin_cleanup_proven":False,"monotonic_s":time.monotonic(),"model_calls":0,
            "native_action_count":None if env is None else list(env.take_action_cnt),
            "benchmark_attempt_started":session is not None and session.lifecycle._claim is not None,
            "terminal_sha256":file_sha256(terminal_path) if terminal_path.is_file() else None,
            "preparation_sha256":file_sha256(output/"preparation.json"),
            "worker_isolation_sha256":file_sha256(output/"worker_isolation.json"),
            "asset_fidelity_passed":False,"dependency_closure_complete":False}
        for name in ("private_path_preparation","runtime_paths","native_logging_paths",
                     "loaded_module_provenance","loaded_graphics_library_paths"):
            path = output/(name+".json")
            evidence[name+"_sha256"] = file_sha256(path) if path.is_file() else None
        write_durable_json(output/"worker_preclose_ready.json",evidence)
        if app is not None:
            try:
                app.close()  # Stock fast shutdown may exit here after durable evidence.
                write_durable_json(output/"worker_close_returned.json",{"app_close_returned":True,"shutdown_mode":"stock-fast"})
            except BaseException as error:
                write_durable_json(output/"worker_close_failure.json",{"type":type(error).__name__})
                ready = False
    return 0 if ready else 1
