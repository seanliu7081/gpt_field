"""agent_SpeedUp_v2 task registry (spec §6.3).

``V2TaskSpec(key, suite, task_id, prompt, scene, frozen_reset_sha256)``: the official LIBERO sentence and the
frozen reset observation sha256 per measured episode. It holds no targets, regions, mount facts or anything
else a policy could branch on. The policy sees nothing from it; the client sees only the prompt, which the
runner reads from the reset observation after checking it against the simulator's ``list_tasks()``.

Unmeasured episodes stay fail-closed: the runner refuses an episode without a frozen hash unless the operator
passes ``--expected-initial-observation-sha256``.

Hashes: v1's recipe (sha256 of ``json.dumps(obs.to_dict(), sort_keys=True, separators=(",", ":"))``) under the
runner's simulator configuration (seed 7, settle 10, native 512 render and policy images, state8). bowl_plate
and drawer_bowl are v1's frozen values (``scripts/evaluate_agent_speedup_v1.py:142-145``); the libero_10 values
were measured in Phase A and re-measured in M0; the libero_goal values were measured in M0
(``scripts/agent_speedup_v2_measure_resets.py``; record under ``runs/agent_speedup_v2_sim_smoke_m0_resets``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from types import MappingProxyType

TASKS_VERSION = "agent-speedup-v2-tasks-v1"
DEFAULT_SIM_PYTHON = "/workspace/venvs/libero-plus/bin/python"
DEFAULT_LIBERO_ROOT = "/workspace/heading_flow_pub/third_party/LIBERO"
IMAGE_SIZE = 512


@dataclass(frozen=True)
class V2TaskSpec:
    key: str
    suite: str
    task_id: int
    prompt: str
    scene: str
    frozen_reset_sha256: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "frozen_reset_sha256", MappingProxyType(dict(self.frozen_reset_sha256)))

    def to_dict(self):
        return {"key": self.key, "suite": self.suite, "task_id": self.task_id, "prompt": self.prompt,
                "scene": self.scene, "frozen_reset_sha256": {str(k): v for k, v in self.frozen_reset_sha256.items()}}


def _spec(key, suite, task_id, prompt, scene, sha0=None):
    return V2TaskSpec(key, suite, task_id, prompt, scene, {0: sha0} if sha0 else {})


_L10 = (
    ("put both the alphabet soup and the tomato sauce in the basket", "LIVING_ROOM_SCENE2",
     "f9fbbbd914dd6b1181185f5cd543c88742a9c2174ee849a016dc7446186a75ec"),
    ("put both the cream cheese box and the butter in the basket", "LIVING_ROOM_SCENE2",
     "e92ca405b5cfca11e74b915a9ae3c86abcc307ba7847377d1d6d83bc3c25d9cc"),
    ("turn on the stove and put the moka pot on it", "KITCHEN_SCENE3",
     "7b854d9bd19a0d91a843980142f15b6586416ff51790e447b044d6c89d1eceef"),
    ("put the black bowl in the bottom drawer of the cabinet and close it", "KITCHEN_SCENE4",
     "dac6a0409193f4a5f66eb6845579633a693d49c741d2426f6a350dd172c41358"),
    ("put the white mug on the left plate and put the yellow and white mug on the right plate", "LIVING_ROOM_SCENE5",
     "f2a4645e9e1e8b9f18f2e65f1eced6bb9030863cd4b0759541fb1d5e5fbf237f"),
    ("pick up the book and place it in the back compartment of the caddy", "STUDY_SCENE1",
     "a3efc19d471093640e59183b62a2968fd3bc97f8d109ae0f5a8d94b2614876b2"),
    ("put the white mug on the plate and put the chocolate pudding to the right of the plate", "LIVING_ROOM_SCENE6",
     "59d7c04f412daf02dc8818807d8343bda70e6f4dcffa2dfdb16cf9cb35bd048e"),
    ("put both the alphabet soup and the cream cheese box in the basket", "LIVING_ROOM_SCENE1",
     "5ed858a8309893828ce7646d9187890c96345bfd9f5354e0f2815fe6ff565c2c"),
    ("put both moka pots on the stove", "KITCHEN_SCENE8",
     "d07133dc3b5ced57c573b9148abfc1416f66e38b91618f8b0e83b09962cbc4bb"),
    ("put the yellow and white mug in the microwave and close it", "KITCHEN_SCENE6",
     "8fcb564920ae9fe1a37d775922d53e5df515e89d924133516c3158b13920c1a5"),
)
_GOAL = (
    ("open the middle drawer of the cabinet",
     "18013e98a27031fc5f5b787df070b98e5a8df3a2fba1f25702c3a6096842a2fa"),
    ("put the bowl on the stove",
     "3479ba003a1aa9544e9d3cbabbfd93681d2fe1ed23494a1d8e7b556449d19e42"),
    ("put the wine bottle on top of the cabinet",
     "25dee3471c1ba5696a738f0f74801bfbaf8865fd37a19157146db417857f9773"),
    ("open the top drawer and put the bowl inside", "c6853a2d01f24753fce9fe3b06dd05f78d5d667c258fd048144c908f29321eb8"),
    ("put the bowl on top of the cabinet",
     "39a353790a644d2abb4dfe8fcbdd228d23a222f882c98a9d396012d6d0ee1efa"),
    ("push the plate to the front of the stove",
     "6d0d7a1784463127f89a0819c7b90e88f7500926495d1116ac034ccf459a115f"),
    ("put the cream cheese in the bowl",
     "65563db8037b1499b50a3daed92ecb50fd152a2affa367eef5f22454e6cb3732"),
    ("turn on the stove",
     "304487f1fe0547576b2c8c298947db1003f172a577b65446be197eead9519ef8"),
    ("put the bowl on the plate",
     "c08cbab4266c1aac975f36b53e9f83e1ae8777e6f68bde109d98fce7d7922400"),
    ("put the wine bottle on the rack",
     "2bd5022372ecacb520e929c22042eb369ca37e6be2f57b15409ee854f7e8a36b"),
)
_GOAL_SCENE = "LIBERO_GOAL_TABLETOP"

_entries = {
    "bowl_plate": _spec("bowl_plate", "libero_spatial", 0,
                        "pick up the black bowl between the plate and the ramekin and place it on the plate",
                        "LIBERO_SPATIAL_TABLETOP", "8626d3e51bf8618a2c2793dbac9121dc8c87106197b95a7a6e7f483ffed65835"),
    "drawer_bowl": _spec("drawer_bowl", "libero_goal", 3, _GOAL[3][0], _GOAL_SCENE, _GOAL[3][1]),
}
for _i, (_prompt, _scene, _sha) in enumerate(_L10):
    _entries[f"libero_10_t{_i}"] = _spec(f"libero_10_t{_i}", "libero_10", _i, _prompt, _scene, _sha)
for _i, (_prompt, _sha) in enumerate(_GOAL):
    if _i != 3:
        _entries[f"libero_goal_g{_i}"] = _spec(f"libero_goal_g{_i}", "libero_goal", _i, _prompt, _GOAL_SCENE, _sha)
_TASKS = MappingProxyType(_entries)
V2_TASK_ALIASES = MappingProxyType({"libero_goal_g3": "drawer_bowl"})
V2_TASK_KEYS = tuple(_TASKS)
PRIMARY_TASK = "libero_10_t2"
REGRESSION_TASKS = ("bowl_plate", "drawer_bowl")
BACKUP_TASK = "libero_10_t7"
# v1's own tasks keep v1's task_manipulation_v1 monitor chained (record-only) for comparability (spec §6.4).
_V1_MONITOR_KEYS = frozenset(REGRESSION_TASKS)


def resolve_v2_task(task_key):
    if not isinstance(task_key, str):
        raise ValueError("task key must be a string")
    task_key = V2_TASK_ALIASES.get(task_key, task_key)
    if task_key not in _TASKS:
        raise ValueError(f"Unknown agent_speedup_v2 task {task_key!r}; choose one of {', '.join(V2_TASK_KEYS)}")
    return _TASKS[task_key]


def list_v2_tasks():
    return [_TASKS[key] for key in V2_TASK_KEYS]


def frozen_reset_sha256(task_key, episode_index):
    """The frozen hash, or None (fail-closed: the runner then requires an explicit expected hash)."""
    return resolve_v2_task(task_key).frozen_reset_sha256.get(episode_index)


def collision_base_mode(task_key):
    return "task_manipulation_v1" if resolve_v2_task(task_key).key in _V1_MONITOR_KEYS else "none"


def build_v2_sim_config(task_key, *, sim_python=DEFAULT_SIM_PYTHON, libero_root=DEFAULT_LIBERO_ROOT):
    """v1's eval simulator settings (seed 7, settle 10, native 512, state8, no horizon, record-only contacts).

    Returns (LiberoConfig, monitor_options)."""
    from .libero import LiberoConfig
    spec = resolve_v2_task(task_key)
    config = LiberoConfig(suite=spec.suite, seed=7, max_steps=None, settle_steps=10,
                          libero_root=str(Path(libero_root).expanduser().resolve()),
                          python_executable=os.path.abspath(os.path.expanduser(sim_python)),
                          include_arm_joints=False, render_size=IMAGE_SIZE, image_size=IMAGE_SIZE,
                          collision_check_mode=collision_base_mode(task_key), stop_on_collision=False,
                          stop_on_success=True)
    return config, {"goal_aware": True, "truth_window": True}


def prompt_table():
    return {key: _TASKS[key].prompt for key in V2_TASK_KEYS}
