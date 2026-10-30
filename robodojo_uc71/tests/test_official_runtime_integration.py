"""Synthetic production-wiring checks, never native readiness certification.

Fixtures use fake robot/IK measurements, mock model transport, socket pairs and
temporary synthetic approvals. No native simulator, cuRobo or live model runs.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

from robodojo_uc71.approval import canonical_sha256, file_sha256
from robodojo_uc71.client import NativeExecutiveClient, executive_instructions
from robodojo_uc71.controller import ControllerConfig
from robodojo_uc71.native_config import (CLIENT_MODULE, PLANNER_MODULE,
    DisabledPreviewPlanner, ResetOnlyClient, install_preview_adapters,
    validate_native_planners, validate_native_preview_config)
from robodojo_uc71.native_episode_runner import (EpisodeRunnerError,
    NativeEpisodeRunner, execution_configuration)
from robodojo_uc71.native_runtime import (NativeExecutionGuard, NativeRuntimeError,
    runtime_program_files, validate_official_module_provenance)
from robodojo_uc71.native_worker_client import NativeWorkerClient
from robodojo_uc71.official_control import EE_ACTION_KEYS, OFFICIAL_EE
from robodojo_uc71.schema import ExecutionLimits, validate_controller_capabilities
from client_fixtures import TASK, OfflineTransport, current_fields, frame, plan
from test_fragment_executor import decision, monitoring, short_motion
from test_native_episode import OfficialNativeEnv
from test_native_episode_runner import Harness, OfflineModel
from test_native_worker_protocol import Case


def official_capabilities():
    # Explicitly synthetic candidate settings; never a measured qualification.
    return {"version": "robodojo-uc71-official-ee-controller-1",
            "status": "candidate_unvalidated", **monitoring().checked()}


class OfficialHarness(Harness):
    def __init__(self, tmp_path, *, frozen_offline_client=False, make_runner=True):
        self.case = Case(tmp_path, env_type=OfficialNativeEnv, control_backend=OFFICIAL_EE)
        self.model = OfflineModel()
        self.model.controller_capabilities = official_capabilities()
        execution = execution_configuration(controller_config=ControllerConfig(), monitoring=monitoring(),
            limits=self.model.limits, initial_gripper_commands={"left": "open", "right": "open"},
            controller_capabilities=self.model.controller_capabilities, control_backend=OFFICIAL_EE)
        self.case.manifest["configuration"]["execution"] = execution
        self.case.manifest.pop("manifest_sha256")
        self.case.manifest["manifest_sha256"] = canonical_sha256(self.case.manifest)
        self.case.session.manifest = deepcopy(self.case.manifest)
        self.alive = True
        self.client = NativeWorkerClient(self.case.parent, manifest=self.case.manifest,
            expected_worker=self.case.identity, verify_tracked_worker=lambda identity: self.alive and identity == self.case.identity,
            initial_gripper_commands={"left": "open", "right": "open"}, control_backend=OFFICIAL_EE)
        if frozen_offline_client:
            self.transport = OfflineTransport()
            self.model = NativeExecutiveClient(task_prompt=TASK, approved_plan=plan(), direction_frame=frame(),
                transport=self.transport, log_dir=self.client.output / "subscription_logs",
                controller_capabilities=self.model.controller_capabilities, limits=ExecutionLimits())
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.worker = self.pool.submit(self.serve)
        self.runner = self.make_runner() if make_runner else None

    def make_runner(self):
        return NativeEpisodeRunner(self.client, model_client=self.model, monitoring=monitoring(),
            controller_config=ControllerConfig(), limits=self.model.limits)


@pytest.fixture
def official_harness(tmp_path):
    value = OfficialHarness(tmp_path)
    try:
        yield value
    finally:
        value.cleanup()


def test_parent_runner_routes_final_pose_to_native_ee_and_counts_each_action(official_harness):
    harness = official_harness
    harness.model.answers = [short_motion(steps=3), decision(kind="observe")]
    harness.begin()
    harness.runner.run_next()
    harness.runner.run_next()
    env = harness.case.env
    assert harness.runner._executor.control_backend == OFFICIAL_EE
    assert len(env.actions) == env.reward_manager.calls == env.terminal_calls == 3
    assert env.live_steps == 30 and harness.client.lifecycle.confirmed_actions == 3
    assert harness.case.session.lifecycle.confirmed_actions == 3
    assert len(env.ik_calls) == 6
    assert all(set(action) == EE_ACTION_KEYS for action in env.actions)
    # DLS would emit a smaller, changing joint target; the official path gets
    # the complete model target on every native EEF dispatch.
    assert all(action["left_ee_pose"] == [.04, 0., 0., 1., 0., 0., 0.] for action in env.actions)
    assert all(action["right_ee_pose"] == [0., 0., 0., 1., 0., 0., 0.] for action in env.actions)
    frames = sorted((harness.client.output / "video_frames").glob("*.json"))
    assert [json.loads(p.read_text())["native_action_index"] for p in frames] == [0, 1, 2, 3]
    for _, feedback, execution in harness.model.requests:
        text = json.dumps({"feedback": feedback, "execution": execution})
        assert "native_private_reason" not in text and "native_reward" not in text


@pytest.mark.parametrize("failed_arm", ["left", "right"])
def test_parent_runner_reports_ik_failure_after_gripper_and_native_accounting(official_harness, failed_arm):
    harness = official_harness
    harness.case.env.ik_status[failed_arm] = "Fail"
    harness.model.answers = [short_motion(steps=3, grip="close"), decision(kind="observe")]
    harness.begin()
    harness.runner.run_next()
    harness.runner.run_next()
    env = harness.case.env
    assert len(env.actions) == env.take_action_cnt[0] == 1
    assert harness.client.lifecycle.confirmed_actions == harness.case.session.lifecycle.confirmed_actions == 1
    assert env.gripper_targets["left"] == [0.]
    feedback = harness.model.requests[1][1]
    assert feedback["reason"] == "ik_unavailable" and feedback["executed_steps"] == 1
    assert feedback["gripper_command"] == "close"
    assert "ik_status" not in feedback and "native_private_reason" not in json.dumps(feedback)


def test_official_terminal_record_and_final_frame_survive_native_success(official_harness):
    harness = official_harness
    harness.model.answers = [short_motion(steps=3)]
    harness.case.env.reward_manager.reward = 1.
    harness.begin()
    result = harness.runner.run_next()
    assert result["state"] == "TERMINAL"
    assert len(harness.case.env.actions) == 1
    terminal = json.loads((harness.client.output / "terminal.json").read_text())
    assert terminal["official_success"] is True and terminal["confirmed_actions"] == 1
    assert (harness.client.output / "video_frames/frame_0001.json").is_file()


@pytest.mark.parametrize("change", ["legacy_capability", "monitoring", "native_semantics"])
def test_official_runner_rejects_execution_not_exactly_bound_by_manifest(tmp_path, change):
    harness = OfficialHarness(tmp_path, make_runner=False)
    try:
        if change == "legacy_capability":
            harness.model.controller_capabilities = OfflineModel().controller_capabilities
        elif change == "monitoring":
            harness.client.manifest["configuration"]["execution"]["monitoring"]["arrival_stable_samples"] += 1
        else:
            harness.client.manifest["configuration"]["execution"]["native_semantics"]["custom_dls"] = True
        with pytest.raises(EpisodeRunnerError):
            harness.make_runner()
        assert not harness.case.env.actions and not harness.model.requests
    finally:
        harness.cleanup()


def test_production_executive_with_offline_transport_uses_official_prompt(tmp_path):
    harness = OfficialHarness(tmp_path, frozen_offline_client=True)
    try:
        harness.begin()
        harness.runner.run_next()
        assert len(harness.transport.calls) == 1 and not harness.case.env.actions
        payload = harness.transport.calls[0]
        text = json.dumps(payload)
        assert "official cuRobo IK" in text
        assert "bounded measured-error joint targets" not in text
        assert "declared bounded robot-only controller" not in text
        fields = current_fields(payload)
        assert "native_reward" not in fields and "ik_status" not in fields
        assert set(fields["robot"]) == {"left", "right"}
        assert not {"action_translation_m", "action_rotation_rad", "max_joint_step_rad",
                    "joint_target_lead_rad"} & harness.model.controller_capabilities.keys()
    finally:
        harness.cleanup()


def test_official_prompt_and_capabilities_disclose_native_behavior():
    text = executive_instructions(control_backend=OFFICIAL_EE)
    assert "Both arms undergo IK" in text and "fixed table model" in text
    assert "IK can change its joint configuration" in text
    assert "gripper action still executes" in text
    assert "do not impose different Cartesian execution speeds" in text
    for field, value in (("action_translation_m", {"transit": .02, "contact": .005}),
                         ("action_rotation_rad", .1), ("max_joint_step_rad", .05),
                         ("joint_target_lead_rad", .05)):
        with pytest.raises(ValueError, match="does not inherit"):
            validate_controller_capabilities({**official_capabilities(), field: value})
    assert "bounded measured-error joint targets" in executive_instructions()


def test_official_configuration_leaves_native_planner_module_untouched():
    sentinel = ModuleType(PLANNER_MODULE)
    sentinel.CuroboPlanner = object()
    with patch.dict(sys.modules):
        sys.modules.pop(CLIENT_MODULE, None)
        sys.modules[PLANNER_MODULE] = sentinel
        result = install_preview_adapters(control_backend=OFFICIAL_EE)
        assert sys.modules[PLANNER_MODULE] is sentinel
        assert sys.modules[CLIENT_MODULE].WsModelClient is ResetOnlyClient
        assert "native CuroboPlanner" in result[PLANNER_MODULE]
    with patch.dict(sys.modules):
        sys.modules.pop(CLIENT_MODULE, None)
        sys.modules.pop(PLANNER_MODULE, None)
        install_preview_adapters(control_backend=OFFICIAL_EE)
        assert PLANNER_MODULE not in sys.modules


def test_official_configuration_rejects_stale_disabled_planner():
    with patch.dict(sys.modules):
        sys.modules.pop(CLIENT_MODULE, None)
        sys.modules.pop(PLANNER_MODULE, None)
        install_preview_adapters()
        assert sys.modules[PLANNER_MODULE].CuroboPlanner is DisabledPreviewPlanner
        sys.modules.pop(CLIENT_MODULE)
        with pytest.raises(RuntimeError, match="disabled preview planner"):
            install_preview_adapters(control_backend=OFFICIAL_EE)


def native_config(backend):
    return OmegaConf.create({"eval_cfg": {"task_name": "make_toast", "config_name": "arx_x5",
        "seed": 0, "num_envs": 1, "eval_batch": False}, "sim": {"scene": {"num_envs": 1},
        "seed": [0], "device": "cpu", "use_fabric": False, "dt": .004, "decimation": 1,
        "render_interval": 10, "physx": {"enable_stabilization": True}},
        "robot": {"robots": [{"robot_name": "x5", "need_planner": backend == OFFICIAL_EE,
                               "enabled_self_collisions": False} for _ in range(2)]}})


def test_native_config_requires_both_selected_planners_and_retains_physics():
    official = native_config(OFFICIAL_EE)
    validate_native_preview_config(official, control_backend=OFFICIAL_EE)
    with pytest.raises(ValueError):
        validate_native_preview_config(official)
    official.robot.robots[1].need_planner = False
    with pytest.raises(ValueError):
        validate_native_preview_config(official, control_backend=OFFICIAL_EE)
    legacy = native_config("legacy_joint")
    validate_native_preview_config(legacy)
    with pytest.raises(ValueError):
        validate_native_preview_config(legacy, control_backend=OFFICIAL_EE)


def test_selected_native_planner_contract_rejects_disabled_or_replaced_instances():
    # Only exercise the identity contract; this fake is not a native solver.
    planner_type = type("CuroboPlanner", (), {"__module__": PLANNER_MODULE,
        "solve_ik_to_joint": lambda self, *args: None})
    planner = planner_type()
    manager = NS(robot_list=[NS(type="target", robot_name="x5")],
                 planner={"x5": planner}, ik_solver={"x5": planner})
    env = NS(robot_manager=manager)
    validate_native_planners(env, OFFICIAL_EE)
    with pytest.raises(ValueError, match="disabled_native_planners"):
        validate_native_planners(env)
    manager.ik_solver = {"x5": planner_type()}
    with pytest.raises(ValueError, match="unmodified_native_planner"):
        validate_native_planners(env, OFFICIAL_EE)
    manager.planner = manager.ik_solver = {}
    validate_native_planners(env)
    with pytest.raises(ValueError, match="all_native_ik_solvers"):
        validate_native_planners(env, OFFICIAL_EE)


def test_native_guard_backend_must_match_bound_session(tmp_path):
    case = Case(tmp_path, env_type=OfficialNativeEnv, control_backend=OFFICIAL_EE)
    env = case.env
    env.reset = lambda seed: env.sim_step()
    env.take_action_batch = env.step = lambda *args, **kwargs: None
    try:
        guard = NativeExecutionGuard(env, control_backend="legacy_joint")
        guard.reset_once()
        with pytest.raises(NativeRuntimeError, match="guard_backend_differs"):
            guard.bind(case.session)
        assert not env.actions
    finally:
        case.close_sockets()


def test_approved_runtime_source_closure_binds_official_action_conversion():
    assert "official_control.py" in {path.name for path in runtime_program_files()}


@pytest.fixture
def synthetic_module_provenance(tmp_path):
    overlay = tmp_path / "synthetic_overlay"
    modules = {}
    for name in ("curobo", "warp"):
        path = overlay/name/"__init__.py"
        path.parent.mkdir(parents=True)
        path.write_text(f"# Synthetic {name} entry point; never imported.\n")
        modules[name] = {"path": str(path), "sha256": file_sha256(path)}
    return {"overlay_target": str(overlay), "installed_files": list(deepcopy(modules).values())}, modules


def test_official_module_provenance_accepts_exact_bound_package_entry_points(synthetic_module_provenance):
    assert validate_official_module_provenance(*synthetic_module_provenance) is True


@pytest.mark.parametrize("mutation", ["swap", "duplicate", "outside", "tampered", "wrong_hash",
    "missing_installed", "symlink_file", "symlink_parent", "unrelated_installed_file"])
def test_official_module_provenance_rejects_mislabeled_or_changed_sources(synthetic_module_provenance, tmp_path, mutation):
    setup, modules = synthetic_module_provenance
    if mutation == "swap":
        modules["curobo"], modules["warp"] = modules["warp"], modules["curobo"]
    elif mutation == "duplicate":
        modules["warp"] = deepcopy(modules["curobo"])
    elif mutation in ("outside", "unrelated_installed_file"):
        path = tmp_path/"outside/__init__.py" if mutation == "outside" else Path(setup["overlay_target"])/"other.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Synthetic unrelated module, never imported.\n")
        modules["curobo"] = {"path": str(path), "sha256": file_sha256(path)}
        setup["installed_files"].append(deepcopy(modules["curobo"]))
    elif mutation == "tampered":
        Path(modules["warp"]["path"]).write_text("# Changed after binding.\n")
    elif mutation == "wrong_hash":
        modules["warp"]["sha256"] = "f" * 64
    elif mutation == "missing_installed":
        setup["installed_files"] = setup["installed_files"][:1]
    elif mutation == "symlink_file":
        path = Path(modules["warp"]["path"])
        outside = tmp_path / "same_bytes.py"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path = Path(modules["warp"]["path"]).parent
        outside = tmp_path / "relocated_warp"
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(NativeRuntimeError, match="official_module_provenance_invalid"):
        validate_official_module_provenance(setup, modules)
