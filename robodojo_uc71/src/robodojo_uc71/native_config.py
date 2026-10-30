"""Native configuration and explicit adapters for preview and UC71 execution.

Importing this module starts no simulator, transport, or CUDA work. Native source
files remain untouched. Both backends disable native model transport; only the
legacy joint backend disables cuRobo. The parent UC71 client owns model requests.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

EXPECTED_CAMERAS = frozenset({"cam_head", "cam_left_wrist", "cam_right_wrist"})
CLIENT_MODULE = "client_server.ws.model_client"
PLANNER_MODULE = "env.planner_manager.curobo_planner"


class ResetOnlyClient:
    """A non-network reset sink; no inference, observations, or second reset."""

    def __init__(self, **_transport_arguments):
        self.reset_calls = 0
        self.closed = False

    def call(self, func_name=None, obs=None, **kwargs):
        if self.closed or func_name != "reset" or obs is not None or kwargs or self.reset_calls:
            raise RuntimeError("Preview client permits exactly one reset and no model calls")
        self.reset_calls += 1
        return None

    def close(self):
        self.closed = True


class DisabledPreviewPlanner:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Scene-aware planning is disabled in the joint-mode reset-only preview")


def install_preview_adapters(*, control_backend="legacy_joint") -> dict:
    """Disable native transport and select the declared planner backend."""
    from .official_control import validate_control_backend
    validate_control_backend(control_backend)
    replacements = {
        CLIENT_MODULE: ("WsModelClient", ResetOnlyClient),
    }
    if control_backend == "legacy_joint":
        replacements[PLANNER_MODULE] = ("CuroboPlanner", DisabledPreviewPlanner)
    elif getattr(sys.modules.get(PLANNER_MODULE), "__preview_adapter__", False):
        raise RuntimeError("Official control cannot reuse a disabled preview planner")
    if any(name in sys.modules for name in replacements):
        raise RuntimeError("Preview adapters must precede native evaluator/planner imports")
    for name, (symbol, implementation) in replacements.items():
        module = types.ModuleType(name, "Explicit reset-only preview adapter; native source is unchanged")
        setattr(module, symbol, implementation)
        module.__preview_adapter__ = True
        sys.modules[name] = module
    return {
        CLIENT_MODULE: "ResetOnlyClient: exactly one reset; no transport imported or constructed",
        PLANNER_MODULE: ("native CuroboPlanner: official end-effector IK and scene constraints"
            if control_backend == "official_ee" else
            "DisabledPreviewPlanner: every constructor call raises; need_planner is false"),
    }


def add_native_import_paths(root: Path) -> list[str]:
    paths = [root, root / "XPolicyLab"]
    # Source paths expose the exact pinned packages without editable installation.
    lab = root / "third_party/IsaacLab/source"
    paths.extend(lab / name for name in ("isaaclab", "isaaclab_tasks", "isaaclab_assets"))
    for path in reversed(paths):
        if not path.is_dir():
            raise ValueError(f"Missing pinned import directory: {path}")
        sys.path.insert(0, str(path))
    return [str(path) for path in paths]


def build_native_preview_config(root: Path, run_id: str, *, control_backend="legacy_joint"):
    """Use native loaders/processors, then only constrain scope and planner use."""
    from omegaconf import OmegaConf

    pipeline = importlib.import_module("utils.pipeline_utils")
    loader = importlib.import_module("utils.load_file")
    if not Path(pipeline.__file__).resolve().is_relative_to(root):
        raise RuntimeError("The configuration pipeline is not from the pinned native checkout")
    eval_cfg = loader.load_yaml(str(root / "env_cfg/arx_x5.yml"))
    eval_cfg.update(task_name="make_toast", num_envs=1, device_id=0, eval_batch=False,
                    policy_name="uc71_reset_only_preview", additional_info=run_id,
                    seed=0, physx_monitor_enabled=True)
    selection = eval_cfg["config"]
    parts = {section: loader.load_yaml(str(root / "env_cfg" / section / (selection[section] + ".yml")))
             for section in ("sim", "scene", "camera", "robot")}
    parts["task_env"] = loader.load_yaml(str(root / "task/RoboDojo/config/make_toast.yml"))
    parts["eval_cfg"] = eval_cfg
    # Mandatory native fields are inert: the replacement never imports transport.
    parts["deploy_cfg"] = {"policy_name": "uc71_reset_only_preview", "port": 1,
                           "host": "127.0.0.1", "protocol": "ws",
                           "policy_server_url": "ws://127.0.0.1:1",
                           "evaluation_id": run_id, "trial_id": f"make_toast-{run_id}",
                           "action_case_id": "make_toast_case", "repeat_index": None}
    config = OmegaConf.create(parts)
    config = pipeline.process_randomization(config)
    config, eval_num = pipeline.process_config(config, "make_toast")
    config.eval_cfg.eval_num = eval_num
    config.sim.scene.num_envs = 1
    config.eval_cfg.num_envs = 1
    config.sim.seed = [0]
    config.camera.default_frequency = config.eval_cfg.observation.get("collect_freq", 0)
    for robot in config.robot.robots:
        robot.need_planner = control_backend == "official_ee"
    validate_native_preview_config(config, control_backend=control_backend)
    return config


def validate_native_preview_config(config, *, control_backend="legacy_joint") -> None:
    from .official_control import validate_control_backend
    validate_control_backend(control_backend)
    if (config.eval_cfg.task_name != "make_toast" or config.eval_cfg.config_name != "arx_x5"
            or config.eval_cfg.seed != 0 or config.eval_cfg.num_envs != 1
            or config.eval_cfg.eval_batch is not False or config.sim.scene.num_envs != 1
            or list(config.sim.seed) != [0]):
        raise ValueError("Preview is bound to Standard Make Toast / arx_x5 / one environment / layout 0")
    if config.sim.get("device", "cpu") != "cpu" or config.sim.get("use_fabric", False) is not False:
        raise ValueError("Preserve native CPU physics and disabled fabric")
    if (config.sim.dt != 0.004 or config.sim.decimation != 1 or config.sim.render_interval != 10
            or config.sim.physx.enable_stabilization is not True):
        raise ValueError("Native Make Toast physics/stabilization configuration changed")
    robots = config.robot.robots
    if len(robots) != 2 or any(robot.robot_name != "x5" or robot.get("coupled", False)
                             or robot.need_planner is not (control_backend == "official_ee")
                             or robot.enabled_self_collisions is not False for robot in robots):
        raise ValueError("Require both native X5 robots, selected backend planners, and native self-collision settings")


def forbid_policy_actions(env) -> list[str]:
    def forbidden(*args, **kwargs):
        raise RuntimeError("Policy inference and benchmark actions are forbidden during reset-only preview")
    names = [name for name in ("run_eval", "eval_one_episode", "eval_one_episode_batch",
                              "take_action", "take_action_batch", "step") if hasattr(env, name)]
    for name in names:
        setattr(env, name, forbidden)
    return names


def validate_native_planners(env, control_backend="legacy_joint"):
    """Check the selected native action path without invoking either solver."""
    from .official_control import validate_control_backend
    validate_control_backend(control_backend)
    manager = env.robot_manager
    if control_backend == "legacy_joint":
        if manager.planner or manager.ik_solver:
            raise ValueError("legacy_control_requires_disabled_native_planners")
        return
    names = {r.robot_name for r in manager.robot_list if r.type == "target"}
    if not names or set(manager.planner) != names or set(manager.ik_solver) != names:
        raise ValueError("official_control_requires_all_native_ik_solvers")
    for name in names:
        planner = manager.planner[name]
        if (manager.ik_solver[name] is not planner
                or type(planner).__module__ != PLANNER_MODULE
                or type(planner).__name__ != "CuroboPlanner"
                or not callable(getattr(planner, "solve_ik_to_joint", None))):
            raise ValueError("official_control_requires_unmodified_native_planner")
