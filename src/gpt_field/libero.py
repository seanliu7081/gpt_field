"""Optional LIBERO simulator in a separate process with a JSONL boundary.

Only ``PolicyObservation.to_dict()`` crosses as an observation. Success flags,
task indices, and dependency provenance are returned separately to the evaluator.
The policy process does not import LIBERO, robosuite, MuJoCo, or torch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from .observation import PolicyObservation

SUITE_TASK_COUNTS = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_90": 90, "libero_10": 10}


@dataclass(frozen=True)
class LiberoConfig:
    suite: str = "libero_10"
    seed: int = 7
    image_size: int = 224
    render_size: int = 256
    max_steps: int | None = 520
    settle_steps: int = 10
    libero_root: str | None = None
    python_executable: str | None = None
    response_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.suite not in SUITE_TASK_COUNTS:
            raise ValueError(f"unsupported official LIBERO suite: {self.suite}")
        for name in ("image_size", "render_size", "max_steps", "settle_steps", "seed"):
            value = getattr(self, name)
            if name == "max_steps" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name in {"image_size", "render_size", "max_steps"} else 0):
                raise ValueError(f"{name} has an invalid value")
        if not math.isfinite(self.response_timeout_s) or self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")


class SimulatorError(RuntimeError):
    pass


def _validate_action(action: Sequence[float]) -> list[float]:
    values = list(action)
    if len(values) != 7 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 1 for v in values):
        raise ValueError("simulator action must contain 7 finite normalized numbers in [-1, 1]")
    return [float(v) for v in values]


class SimProcess:
    """Synchronous, timeout-bounded client; ``command`` supports test transports."""

    def __init__(self, config: LiberoConfig, *, command: Sequence[str] | None = None):
        self.config = config
        self._command = list(command) if command is not None else None
        self._process: subprocess.Popen | None = None
        self._buffer = bytearray()
        self.metadata: dict[str, Any] = {}
        self.status: dict[str, Any] = {}

    def start(self) -> "SimProcess":
        if self._process is not None:
            return self
        command = self._command or [
            self.config.python_executable or sys.executable, "-u", "-m", "gpt_field.libero",
            "--worker", "--config-json", json.dumps(asdict(self.config)),
        ]
        env = os.environ.copy()
        package_src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_src, env.get("PYTHONPATH")]))
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0)
        try:
            self.metadata = self._request({"op": "metadata"})
        except BaseException:
            self.close()
            raise
        return self

    def _request(self, request: dict) -> dict:
        if self._process is None:
            raise SimulatorError("simulator process has not been started")
        proc = self._process
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SimulatorError("simulator worker closed its input") from exc
        import time
        deadline = time.monotonic() + self.config.response_timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    # A late response must never be mistaken for the next request.
                    self.close()
                    raise SimulatorError("simulator response timed out; worker terminated")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise SimulatorError(f"simulator worker exited without a response (status {proc.poll()})")
                self._buffer.extend(chunk)
                if len(self._buffer) > 16 * 1024 * 1024:
                    self.close()
                    raise SimulatorError("simulator response exceeds transport limit")
        line, _, tail = self._buffer.partition(b"\n")
        self._buffer = bytearray(tail)
        try:
            response = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise SimulatorError("simulator emitted invalid JSON") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            error = response.get("error", "malformed response") if isinstance(response, dict) else "malformed response"
            raise SimulatorError(f"simulator worker: {error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise SimulatorError("simulator response result must be an object")
        return result

    def list_tasks(self) -> list[dict[str, Any]]:
        self.start()
        return self._request({"op": "tasks"})["tasks"]

    def reset(self, task_id: int, episode_index: int) -> PolicyObservation:
        task_count = SUITE_TASK_COUNTS[self.config.suite]
        if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < task_count:
            raise ValueError(f"task_id must be an integer in [0, {task_count - 1}]")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or not 0 <= episode_index < 50:
            raise ValueError("episode_index must be an integer in [0, 49]; no modulo wrapping")
        self.start()
        result = self._request({"op": "reset", "task_id": task_id, "episode_index": episode_index})
        self.status = result["status"]
        return PolicyObservation.from_dict(result["observation"])

    def step(self, action: Sequence[float]) -> tuple[PolicyObservation, bool]:
        values = _validate_action(action)
        if not self.status or self.status.get("terminated"):
            raise SimulatorError("reset is required before stepping an inactive episode")
        result = self._request({"op": "step", "action": values})
        self.status = result["status"]
        return PolicyObservation.from_dict(result["observation"]), bool(self.status["success"])

    def close(self) -> None:
        proc, self._process = self._process, None
        self.status = {}
        self._buffer.clear()
        if proc is None:
            return
        if proc.stdin is not None:
            proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    def __enter__(self) -> "SimProcess":
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(root: Path) -> dict:
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    revision = None
    # Never mistake an enclosing project's SHA for the vendored LIBERO SHA.
    if (root / ".git").exists():
        completed = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if completed.returncode == 0:
            revision = completed.stdout.strip()
    return {"root": str(root), "git_commit": revision, "python_tree_sha256": digest.hexdigest(), "python_files": len(files)}


class _LiberoWorker:
    """Evaluator-side object. Never instantiate in the model policy process."""

    def __init__(self, config: LiberoConfig):
        self.config = config
        self.env = None
        self.status: dict = {}
        self.prompt = ""
        raw_root = config.libero_root or os.environ.get("LIBERO_ROOT")
        if not raw_root:
            raise ValueError("pass --libero-root pointing to an official LIBERO checkout; no implicit LIBERO-plus fallback")
        root = Path(raw_root).expanduser().resolve()
        if "libero-plus" in str(root).lower() or "libero_plus" in str(root).lower():
            raise ValueError("LIBERO-plus is a distinct benchmark; use an official LIBERO source checkout")
        benchmark_root = root / "libero" / "libero"
        for name in ("benchmark/__init__.py", "envs/env_wrapper.py", f"bddl_files/{config.suite}", f"init_files/{config.suite}"):
            if not (benchmark_root / name).exists():
                raise ValueError(f"LIBERO checkout missing {name}")
        self.root, self.benchmark_root = root, benchmark_root
        self._temp = tempfile.TemporaryDirectory(prefix="gpt-field-libero-")
        paths = {"benchmark_root": str(benchmark_root), "bddl_files": str(benchmark_root / "bddl_files"),
                 "init_states": str(benchmark_root / "init_files"), "assets": str(benchmark_root / "assets"),
                 "datasets": str(root / "libero" / "datasets")}
        # JSON is YAML-compatible and avoids importing LIBERO's interactive setup.
        Path(self._temp.name, "config.yaml").write_text(json.dumps(paths), encoding="utf-8")
        os.environ["LIBERO_CONFIG_PATH"] = self._temp.name
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path.insert(0, str(root))
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite
        import robosuite.controllers.osc as osc
        import robosuite.models.grippers.panda_gripper as panda_gripper
        import robosuite.utils.control_utils as control_utils
        import numpy as np

        if not Path(benchmark.__file__).resolve().is_relative_to(root):
            raise RuntimeError("loaded LIBERO module is outside requested source checkout")
        self._env_class = OffScreenRenderEnv
        self.benchmark = benchmark.get_benchmark_dict()[config.suite](task_order_index=0)
        if self.benchmark.get_num_tasks() != SUITE_TASK_COUNTS[config.suite]:
            raise RuntimeError("unexpected task count in requested official LIBERO suite")
        controller = robosuite.load_controller_config(default_controller="OSC_POSE")
        expected = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        if (controller.get("type") != "OSC_POSE" or controller.get("control_delta") is not True
                or controller.get("impedance_mode") != "fixed"
                or not np.allclose(controller.get("input_max"), 1)
                or not np.allclose(controller.get("input_min"), -1)
                or not np.allclose(controller.get("output_max"), expected)
                or not np.allclose(controller.get("output_min"), -np.array(expected))):
            raise RuntimeError("installed OSC_POSE configuration does not match verified action adapter")
        versions = {}
        for package in ("robosuite", "mujoco", "torch", "numpy", "Pillow"):
            versions[package] = importlib.metadata.version(package)
        controller_files = {"osc": Path(osc.__file__), "control_utils": Path(control_utils.__file__),
                            "panda_gripper": Path(panda_gripper.__file__),
                            "osc_pose_config": Path(robosuite.__file__).parent / "controllers/config/osc_pose.json"}
        self.metadata = {
            "protocol": "gpt-field-libero-jsonl-v1", "benchmark_variant": "LIBERO official layout (explicit checkout)",
            "libero_source": _source_manifest(root), "packages": versions,
            "python_executable": sys.executable, "suite": config.suite, "task_order_index": 0,
            "task_count": self.benchmark.get_num_tasks(),
            "seed": config.seed, "initial_states_per_task": 50, "max_policy_steps": config.max_steps,
            "settle_steps": config.settle_steps, "image_size": config.image_size,
            "render_size": config.render_size,
            "image_protocol": "render RGB -> rotate 180 degrees -> bilinear resize with centered black padding",
            "mujoco_gl": os.environ["MUJOCO_GL"], "controller": controller,
            "controller_sources": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in controller_files.items()},
            "action_order": ["dx/0.05", "dy/0.05", "dz/0.05", "world_rotvec_x/0.5", "world_rotvec_y/0.5", "world_rotvec_z/0.5", "gripper_minus1_open_plus1_close"],
            "success_source": "OffScreenRenderEnv.check_success()", "control_freq_hz": 20,
        }

    def tasks(self) -> list[dict]:
        return [{"task_id": i, "name": self.benchmark.get_task(i).name, "prompt": self.benchmark.get_task(i).language}
                for i in range(self.benchmark.get_num_tasks())]

    def reset(self, task_id: int, episode_index: int) -> dict:
        import numpy as np
        import random
        import torch

        if not 0 <= task_id < self.benchmark.get_num_tasks() or not 0 <= episode_index < 50:
            raise ValueError("task or official episode index out of range")
        if self.env is not None:
            self.env.close()
            self.env = None
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        task = self.benchmark.get_task(task_id)
        self.prompt = task.language
        init_path = self.benchmark_root / "init_files" / task.problem_folder / task.init_states_file
        # Official local .init files contain numpy arrays; torch >=2.6 otherwise
        # defaults to weights_only=True. This is explicit trusted checkout data.
        initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
        if len(initial_states) != 50:
            raise ValueError(f"expected 50 official initial states, found {len(initial_states)}")
        bddl_path = Path(self.benchmark.get_task_bddl_file_path(task_id))
        # robosuite compares timestep >= horizon even with ignore_done=True.
        # Infinity is runtime-only; the serialized configuration retains null.
        horizon = (math.inf if self.config.max_steps is None
                   else self.config.max_steps + self.config.settle_steps + 1)
        self.env = self._env_class(
            bddl_file_name=str(bddl_path), camera_heights=self.config.render_size,
            camera_widths=self.config.render_size, control_freq=20,
            horizon=horizon,
            ignore_done=True, controller="OSC_POSE", robots=["Panda"],
        )
        self.env.seed(self.config.seed)
        self.env.reset()
        obs = self.env.set_init_state(initial_states[episode_index])
        for _ in range(self.config.settle_steps):
            obs, _reward, _done, _info = self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        success = bool(self.env.check_success())
        self.status = {"task_id": task_id, "episode_index": episode_index, "steps": 0,
                       "success": success, "terminated": success, "horizon_reached": False,
                       "initial_state_file": str(init_path), "initial_state_sha256": _sha256(init_path),
                       "bddl_sha256": _sha256(bddl_path)}
        return self._response(obs)

    def step(self, action: Sequence[float]) -> dict:
        if self.env is None or self.status.get("terminated"):
            raise RuntimeError("reset is required before stepping")
        obs, _reward, _done, _info = self.env.step(_validate_action(action))
        success = bool(self.env.check_success())
        self.status["steps"] += 1
        self.status["success"] = success
        self.status["horizon_reached"] = (self.config.max_steps is not None
                                          and self.status["steps"] >= self.config.max_steps)
        self.status["terminated"] = success or self.status["horizon_reached"]
        return self._response(obs)

    def _response(self, obs: dict) -> dict:
        observation = PolicyObservation.from_libero(obs, self.prompt, image_size=self.config.image_size)
        return {"observation": observation.to_dict(), "status": self.status.copy()}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self._temp.cleanup()


def _worker_main(config: LiberoConfig) -> int:
    # Preserve a dedicated protocol FD, then divert Python *and native* stdout
    # from imported simulation packages to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if worker is None:
                    worker = _LiberoWorker(config)
                op = request["op"]
                if op == "metadata":
                    result = worker.metadata
                elif op == "tasks":
                    result = {"tasks": worker.tasks()}
                elif op == "reset":
                    result = worker.reset(request["task_id"], request["episode_index"])
                elif op == "step":
                    result = worker.step(request["action"])
                else:
                    raise ValueError(f"unknown operation: {op}")
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(response, allow_nan=False) + "\n")
            protocol.flush()
    finally:
        if worker is not None:
            worker.close()
        protocol.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    return _worker_main(LiberoConfig(**json.loads(args.config_json)))


if __name__ == "__main__":
    raise SystemExit(main())
