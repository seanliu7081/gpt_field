"""Load the custom drawer/bowl + moka/stove task; no policy/model calls.

Use /workspace/venvs/libero-plus/bin/python. The environment uses the official
LIBERO source checkout, not the LIBERO-plus benchmark. See README.md for usage.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

HERE = Path(__file__).resolve().parent
LIBERO_ROOT = Path("/workspace/heading_flow_pub/third_party/LIBERO")
TASK_KEY = "drawer_bowl_moka_stove_right_no_obstacles"
PROMPT = "open the top drawer and place the bowl on top of the cabinet then close the drawer then turn on the stove and place the moka pot on it"
SEED = 7
SETTLE_STEPS = 10
EXTERNAL_FOVY = 70.0
HOLD = [0., 0., 0., 0., 0., 0., -1.]
_CONFIG = None


@contextmanager
def _configuration_directory():
    # LIBERO caches config_file at import. Keep that file alive until process
    # exit, including between successive environment contexts.
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = tempfile.TemporaryDirectory(prefix="drawer-moka-")
    yield _CONFIG.name


@contextmanager
def load_task(*, render_size=512, bddl_path=None, state_path=None, bootstrap=False, ordered=True):
    """Yield (env, raw_observation, prompt) after the frozen seed-7 reset.

    env.step(action) uses stock seven-dimensional OSC_POSE actions. Default
    check_success() checks ordered progress plus all final predicates;
    raw_goal_success() checks only the BDDL final conjunction. ordered=False
    returns the raw simulator for evaluator-only validation. Raw observations include evaluator
    channels: supply only RGB/robot proprioception plus prompt to a policy.
    Always use this context manager to close MuJoCo. Its small temporary config
    lives until process exit because LIBERO caches the filename on import.
    """
    import numpy as np
    import torch

    bddl_path = Path(bddl_path or HERE / "custom_task.bddl").resolve()
    state_path = Path(state_path or HERE / "initial_state.npy").resolve()
    benchmark_root = LIBERO_ROOT / "libero/libero"
    previous_config = os.environ.get("LIBERO_CONFIG_PATH")
    added_path = str(LIBERO_ROOT) not in sys.path
    env = None
    package = previous_package_config = None
    with _configuration_directory() as config_dir:
        paths = {"benchmark_root": str(benchmark_root),
                 "bddl_files": str(benchmark_root / "bddl_files"),
                 "init_states": str(benchmark_root / "init_files"),
                 "assets": str(benchmark_root / "assets"),
                 "datasets": str(LIBERO_ROOT / "libero/datasets")}
        Path(config_dir, "config.yaml").write_text(json.dumps(paths))
        os.environ["LIBERO_CONFIG_PATH"] = config_dir
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        if added_path:
            sys.path.insert(0, str(LIBERO_ROOT))
        try:
            import libero.libero as package
            previous_package_config = (package.libero_config_path, package.config_file)
            package.libero_config_path = config_dir
            package.config_file = str(Path(config_dir, "config.yaml"))
            from libero.libero.envs import OffScreenRenderEnv
            import libero.libero.envs as envs
            if not Path(envs.__file__).resolve().is_relative_to(LIBERO_ROOT):
                raise RuntimeError("LIBERO was imported from a different checkout")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path), camera_heights=render_size,
                camera_widths=render_size, control_freq=20, horizon=math.inf,
                ignore_done=True, controller="OSC_POSE", robots=["Panda"])
            env.seed(SEED)
            obs = env.reset()
            # This all-right custom scene needs a wider view than stock t08.
            # Camera pose and wrist view are unchanged; keep the whole cabinet
            # visible rather than clipping it at the external image boundary.
            env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] = EXTERNAL_FOVY
            env.sim.forward()
            if not bootstrap:
                state = np.load(state_path, allow_pickle=False)
                expected = np.asarray(env.sim.get_state().flatten()).shape
                if state.shape != expected or state.dtype.kind != "f" or not np.isfinite(state).all():
                    raise ValueError("Saved state must be finite floats matching this BDDL model")
                obs = env.set_init_state(state)
                for _ in range(SETTLE_STEPS):
                    obs, _, _, _ = env.step(HOLD)
            prompt = " ".join(env.env.parsed_problem["language_instruction"])
            if ordered and not bootstrap:
                from ordered_success import OrderedTaskEnv
                wrapped = OrderedTaskEnv(env)
                progress = wrapped.reset_progress()
                if not progress["initial_state_valid"]:
                    raise RuntimeError("Frozen reset does not satisfy the custom task's initial conditions")
                yield wrapped, obs, prompt
            else:
                yield env, obs, prompt
        finally:
            if env is not None:
                env.close()
            if previous_package_config is not None:
                package.libero_config_path, package.config_file = previous_package_config
            if previous_config is None:
                os.environ.pop("LIBERO_CONFIG_PATH", None)
            else:
                os.environ["LIBERO_CONFIG_PATH"] = previous_config
            if added_path:
                sys.path.remove(str(LIBERO_ROOT))


if __name__ == "__main__":
    with load_task() as (env, obs, prompt):
        assert prompt == PROMPT
        assert not env.check_success()
        print(json.dumps({"task_key": TASK_KEY, "prompt": prompt,
                          "initial_success": False, "settle_steps": SETTLE_STEPS}))
