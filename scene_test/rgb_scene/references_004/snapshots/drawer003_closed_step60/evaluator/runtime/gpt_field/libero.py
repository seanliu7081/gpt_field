"""Optional LIBERO simulator in a separate process with a JSONL boundary.

Only ``PolicyObservation.to_dict()`` crosses as an observation. Success flags,
task indices, and dependency provenance are returned separately to the evaluator.
The policy process does not import LIBERO, robosuite, MuJoCo, or torch.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from .observation import JOINT_OBSERVATION_KEYS, OBSERVATION_KEYS, PolicyObservation

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
    stop_on_success: bool = True
    initial_state_override: str | None = None
    task_bddl_override: str | None = None
    collision_check_mode: str = "none"
    include_arm_joints: bool = False
    # Inert when monitoring is disabled. False records contacts without changing
    # LIBERO success or interrupting execution; intended for opt-in evaluations.
    stop_on_collision: bool = True

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
        if type(self.stop_on_success) is not bool:
            raise ValueError("stop_on_success must be a boolean")
        if type(self.include_arm_joints) is not bool:
            raise ValueError("include_arm_joints must be a boolean")
        if type(self.stop_on_collision) is not bool:
            raise ValueError("stop_on_collision must be a boolean")
        if self.initial_state_override is not None and (
                not isinstance(self.initial_state_override, str) or not self.initial_state_override.strip()):
            raise ValueError("initial_state_override must be a nonempty path or None")
        if self.task_bddl_override is not None:
            if not isinstance(self.task_bddl_override, str) or not self.task_bddl_override.strip():
                raise ValueError("task_bddl_override must be a nonempty path or None")
            if self.initial_state_override is None:
                raise ValueError("task_bddl_override requires initial_state_override for its model")
        if self.collision_check_mode not in ("none", "bowl_obstacles_v1", "task_manipulation_v1"):
            raise ValueError("collision_check_mode must be none, bowl_obstacles_v1 or task_manipulation_v1")
        if self.collision_check_mode == "bowl_obstacles_v1" and (
                self.suite != "libero_spatial" or self.task_bddl_override is None
                or self.initial_state_override is None):
            raise ValueError("bowl_obstacles_v1 requires a custom LIBERO-Spatial bowl scene and state")
        if self.collision_check_mode == "task_manipulation_v1" and self.suite not in ("libero_spatial", "libero_goal"):
            raise ValueError("task_manipulation_v1 supports LIBERO-Spatial bowl and LIBERO-Goal drawer tasks")


def _load_initial_state_override(path):
    """Load a custom evaluator-only MuJoCo state without pickle deserialization."""
    import numpy as np

    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("initial_state_override must be at most 1 MiB")
    try:
        state = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, EOFError, OSError) as exc:
        raise ValueError("initial_state_override must be a numeric .npy state") from exc
    if not isinstance(state, np.ndarray):
        if hasattr(state, "close"):
            state.close()
        raise ValueError("initial_state_override must be a single .npy array")
    if (state.ndim != 1 or not state.size or state.dtype.kind != "f"
            or state.dtype.itemsize not in (4, 8) or not np.isfinite(state).all()):
        raise ValueError("initial_state_override must be a nonempty finite 1-D floating-point state")
    return state, {"path": str(resolved), "sha256": hashlib.sha256(data).hexdigest(),
                   "shape": list(state.shape), "dtype": str(state.dtype)}


def validate_initial_state_override(path) -> dict:
    """Validate and identify a layout artifact before authentication or execution."""
    return _load_initial_state_override(path)[1]


def _load_task_bddl(path):
    """Read a bounded, non-executable BDDL expression without simulator imports."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if not data or len(data) > 1024 * 1024:
        raise ValueError("task_bddl_override must contain at most 1 MiB of BDDL text")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("task_bddl_override must contain UTF-8 BDDL text") from exc
    tokens = re.findall(r"[()]|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            group = []
            (stack[-1] if stack else roots).append(group)
            stack.append(group)
            if len(stack) > 128:
                raise ValueError("task_bddl_override nesting is too deep")
        elif token == ")":
            if not stack:
                raise ValueError("task_bddl_override has unbalanced parentheses")
            stack.pop()
        else:
            # The official region parser evaluates yaw expressions. Restrict
            # tokens to numeric arithmetic / names and reject calls, quotes,
            # indexing and other Python syntax before that parser is reached.
            if not stack or re.fullmatch(r"[a-z0-9_:+*/.\-]+", token) is None:
                raise ValueError("task_bddl_override contains an unsupported BDDL token")
            stack[-1].append(token)
    if stack or len(roots) != 1 or not roots[0] or roots[0][0] != "define":
        raise ValueError("task_bddl_override must contain one balanced define expression")
    sections = {}
    for section in roots[0][1:]:
        if not isinstance(section, list) or not section or not isinstance(section[0], str) or section[0] in sections:
            raise ValueError("task_bddl_override contains an invalid or repeated section")
        sections[section[0]] = section[1:]
    for required in ("problem", ":domain", ":language", ":objects", ":goal"):
        if not sections.get(required):
            raise ValueError(f"task_bddl_override is missing {required}")
    if (len(sections["problem"]) != 1 or sections[":domain"] != ["robosuite"]
            or not all(isinstance(word, str) for word in sections[":language"])
            or len(sections[":goal"]) != 1 or not isinstance(sections[":goal"][0], list)):
        raise ValueError("task_bddl_override has an invalid task declaration")
    return sections, {"path": str(resolved), "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def validate_task_bddl_override(path) -> dict:
    """Validate file structure and identity before credentials or model launch."""
    return _load_task_bddl(path)[1]


def _bddl_assets(sections):
    assets = {}
    for section in (":objects", ":fixtures"):
        values, names = sections.get(section, []), []
        index = 0
        while index < len(values):
            token = values[index]
            if not isinstance(token, str):
                raise ValueError("BDDL asset declarations must contain names and classes")
            if token != "-":
                names.append(token)
                index += 1
                continue
            if not names or index + 1 >= len(values) or not isinstance(values[index + 1], str):
                raise ValueError("BDDL asset declaration is missing its class")
            for name in names:
                if name in assets:
                    raise ValueError("BDDL asset instance names must be unique")
                assets[name] = values[index + 1]
            names = []
            index += 2
        if names:
            raise ValueError("BDDL custom scene assets require explicit classes")
    return assets


def _bddl_symbols(expression):
    if isinstance(expression, str):
        yield expression
    elif isinstance(expression, list):
        for child in expression:
            yield from _bddl_symbols(child)


def validate_task_bddl_contract(original_path, override_path) -> dict:
    """Preserve the original task and target assets while adding scene objects.

    This comparison does not import LIBERO or expose object state to the policy.
    The worker separately checks the official parser's interpretation.
    """
    original, _ = _load_task_bddl(original_path)
    changed, _ = _load_task_bddl(override_path)
    for key in ("problem", ":domain", ":language", ":goal", ":obj_of_interest"):
        if original.get(key) != changed.get(key):
            raise ValueError(f"Custom BDDL must preserve original {key}")
    old_assets, new_assets = _bddl_assets(original), _bddl_assets(changed)
    targets = set(original.get(":obj_of_interest", [])) | (set(_bddl_symbols(original[":goal"])) & old_assets.keys())
    if not targets or any(name not in old_assets or new_assets.get(name) != old_assets[name] for name in targets):
        raise ValueError("Custom BDDL must preserve target instance names and asset classes")
    return {"language_preserved": True, "goal_preserved": True, "target_assets_preserved": True,
            "target_assets": {name: old_assets[name] for name in sorted(targets)}}


def _validate_official_bddl_contract(original_path, override_path, contract):
    """Use the environment's parser too, solely inside the evaluator process."""
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    original = robosuite_parse_problem(str(original_path))
    changed = robosuite_parse_problem(str(override_path))
    for key in ("problem_name", "language_instruction", "goal_state", "obj_of_interest"):
        if original.get(key) != changed.get(key):
            raise ValueError(f"Official parser found changed BDDL {key}")
    for description in (original, changed):
        assets = {}
        for section in ("objects", "fixtures"):
            for asset_class, names in description[section].items():
                for name in names:
                    if name in assets:
                        raise ValueError("Official parser found duplicate BDDL asset instance")
                    assets[name] = asset_class
        if any(assets.get(name) != kind for name, kind in contract["target_assets"].items()):
            raise ValueError("Official parser found changed target BDDL asset classes")


class SimulatorError(RuntimeError):
    pass


def _validate_action(action: Sequence[float]) -> list[float]:
    values = list(action)
    if len(values) != 7 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 1 for v in values):
        raise ValueError("simulator action must contain 7 finite normalized numbers in [-1, 1]")
    return [float(v) for v in values]


class _ForbiddenCollisionStop(Exception):
    """Stop the current control command at its first forbidden physics contact."""


def _collision_policy_metadata(stop_on_collision):
    return {
        "stop_on_collision": stop_on_collision,
        "collision_policy": "stop_and_disqualify" if stop_on_collision else "record_only",
        "success_qualification": ("libero_goal_and_no_forbidden_contact" if stop_on_collision
                                  else "raw_libero_goal"),
    }


class _BowlObstacleCollisionMonitor:
    """Evaluator-only contact rule, sampled after every actual MuJoCo step.

    This is a contact-pair rule rather than a swept-volume or force test. Static
    scene contacts and robot-internal contacts are outside its scoring scope.
    """

    mode = "bowl_obstacles_v1"
    required_objects = frozenset({"akita_black_bowl_1", "plate_1", "ketchup_1", "milk_1",
                                  "glazed_rim_porcelain_ramekin_1"})
    required_roles = frozenset({"target_bowl", "plate", "table", "robot", "robot_finger"})
    monitored_scope = [
        "robot links and gripper versus scene objects, plate, table and fixtures",
        "target bowl versus obstacles, fixtures and non-finger robot geometry",
        "MuJoCo collision geoms with contact distance <= 0 at every physics substep and final endpoint",
    ]
    allowances = [
        "named Panda finger and finger-pad collision geoms versus target bowl for grasping",
        "target bowl versus main table and target plate support-contact pairs",
    ]
    excluded_scope = [
        "robot-internal and self contacts, including adjacent robot links",
        "scene-scene contacts without the robot or target bowl, including resting supports",
        "positive-distance collision-margin proximity; no contact-force threshold is used",
        "reset settling substeps; the final initial state is checked before policy execution",
    ]

    def __init__(self, env, *, stop_on_collision=True):
        if type(stop_on_collision) is not bool:
            raise ValueError("stop_on_collision must be a boolean")
        self.stop_on_collision = stop_on_collision
        self.env = env
        self.sim = env.sim
        self.core = getattr(env, "env", env)
        self.first_event = None
        self.counts = {"checks": 0, "physics_substeps": 0, "forbidden_contact_points": 0,
                       "allowed_contact_points": 0, "positive_distance_contacts_ignored": 0}
        self._action_index = None
        self._substep = 0
        self._original_step = self.sim.step
        self._geometry = self._map_geometry()

    def _map_geometry(self):
        model = self.sim.model
        objects = self.core.objects_dict
        if not self.required_objects.issubset(objects):
            raise ValueError(f"{self.mode} requires the target bowl, plate and registered task objects")
        roots = {}
        for name, obj in {**objects, **self.core.fixtures_dict}.items():
            role = "target_bowl" if name == "akita_black_bowl_1" else "plate" if name == "plate_1" else "scene"
            roots[int(model.body_name2id(obj.root_body))] = (name, role)
        finger_names = set()
        for index, robot in enumerate(self.core.robots):
            roots[int(model.body_name2id(robot.robot_model.root_body))] = (f"robot{index}", "robot")
            gripper = robot.gripper
            if isinstance(gripper, dict):
                raise ValueError(f"{self.mode} requires the single Panda gripper")
            roots[int(model.body_name2id(gripper.root_body))] = (f"robot{index}_gripper", "robot")
            for key in ("left_finger", "right_finger", "left_fingerpad", "right_fingerpad"):
                finger_names.update(gripper.important_geoms.get(key, []))
        if not finger_names:
            raise ValueError("Cannot identify the intentional Panda finger contact geoms")
        explicit_pairs = set()
        for i in range(int(getattr(model, "npair", 0))):
            explicit_pairs.update((int(model.pair_geom1[i]), int(model.pair_geom2[i])))
        result = {}
        for geom_id in range(model.ngeom):
            if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id] or geom_id in explicit_pairs):
                continue
            name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
            body_id = int(model.geom_bodyid[geom_id])
            body = model.body_id2name(body_id) or f"body_{body_id}"
            current, visited = body_id, set()
            owner, role = body, "scene"
            while current not in visited:
                visited.add(current)
                if current in roots:
                    owner, role = roots[current]
                    break
                if current == 0:
                    break
                current = int(model.body_parentid[current])
            if name == "table_collision":
                owner, role = "main_table", "table"
            if name in finger_names:
                if role != "robot":
                    raise ValueError("Finger collision geometry is outside the robot body tree")
                role = "robot_finger"
            result[geom_id] = {"id": geom_id, "name": name, "body": body, "object": owner, "role": role}
        if not self.required_roles.issubset({x["role"] for x in result.values()}):
            raise ValueError("Collision monitor could not map required robot, target and support geometry")
        return result

    @staticmethod
    def _forbidden(first, second):
        roles = {first["role"], second["role"]}
        robot = {"robot", "robot_finger"}
        if first["role"] in robot and second["role"] in robot:
            return None
        if roles == {"robot_finger", "target_bowl"}:
            return None
        if roles in ({"target_bowl", "table"}, {"target_bowl", "plate"}, {"target_bowl"}):
            return None
        if roles & robot:
            return "robot_nonfinger_target_bowl_contact" if "target_bowl" in roles else "robot_scene_contact"
        if "target_bowl" in roles:
            return "target_bowl_scene_contact"
        return None

    def _contact_reason(self, first, second, contact):
        return self._forbidden(first, second)

    def inspect(self, *, phase, action_index=None, physics_substep=None):
        self.counts["checks"] += 1
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            distance = float(contact.dist)
            if not math.isfinite(distance):
                raise ValueError("Non-finite simulator contact distance")
            if distance > 0:
                self.counts["positive_distance_contacts_ignored"] += 1
                continue
            first, second = self._geometry.get(int(contact.geom1)), self._geometry.get(int(contact.geom2))
            if first is None or second is None:
                continue
            reason = self._contact_reason(first, second, contact)
            if reason is None:
                self.counts["allowed_contact_points"] += 1
                continue
            self.counts["forbidden_contact_points"] += 1
            if self.first_event is None:
                self.first_event = {"phase": phase, "action_index": action_index,
                    "physics_substep": physics_substep, "contact_distance_m": distance,
                    "geom1": copy.deepcopy(first), "geom2": copy.deepcopy(second), "reason": reason}

    def attach(self):
        self.sim.step = self._physics_step

    def begin_action(self, action_index):
        if self.stop_on_collision and self.first_event is not None:
            raise RuntimeError("Cannot execute an action after a forbidden collision")
        self._action_index, self._substep = action_index, 0

    def _physics_step(self, *args, **kwargs):
        if self._action_index is None:
            raise RuntimeError("Monitored physics step is outside an acknowledged action")
        result = self._original_step(*args, **kwargs)
        self.counts["physics_substeps"] += 1
        current = self._substep
        self._substep += 1
        self.inspect(phase="action", action_index=self._action_index, physics_substep=current)
        if self.stop_on_collision and self.first_event is not None:
            raise _ForbiddenCollisionStop()
        return result

    def finish_action(self):
        if self._substep == 0:
            raise RuntimeError("Collision monitor did not observe any actual physics substeps")
        if self.stop_on_collision and self.first_event is not None:
            self._action_index = None
            return
        # mj_step's contact buffer describes the dynamics calculation. Refresh
        # the final integrated configuration too, without advancing simulation.
        self.sim.forward()
        self.inspect(phase="action", action_index=self._action_index, physics_substep=self._substep - 1)
        self._action_index = None

    def snapshot(self):
        return {"enabled": True, "mode": self.mode, "detected": self.first_event is not None,
                **_collision_policy_metadata(self.stop_on_collision),
                "first_event": copy.deepcopy(self.first_event), "counts": dict(self.counts),
                "monitored_scope": list(self.monitored_scope), "allowances": list(self.allowances),
                "excluded_scope": list(self.excluded_scope)}


def _collision_task_key(suite, task_id):
    tasks = {("libero_spatial", 0): "bowl_plate", ("libero_goal", 3): "drawer_bowl"}
    try:
        return tasks[(suite, task_id)]
    except KeyError as exc:
        raise ValueError("task_manipulation_v1 supports only bowl_plate (Spatial 0) and drawer_bowl (Goal 3)") from exc


class _TaskManipulationCollisionMonitor(_BowlObstacleCollisionMonitor):
    """Task-specific contact allowances, independent of a layout or demo.

    Drawer shell / handle roles come from the runtime fixture XML, not from a
    whole-cabinet exemption or geom ordinal guesses. For placement, a bowl-shell
    contact additionally must lie in the named top-drawer containment region.
    All of these names, collision contacts, and region coordinates stay here in
    the evaluator; PolicyObservation contains none of them.
    """

    mode = "task_manipulation_v1"
    drawer_support_tolerance_m = .002

    @classmethod
    def rule_description(cls, task_key):
        if task_key not in ("bowl_plate", "drawer_bowl"):
            raise ValueError("unsupported collision task")
        allowances = ["named Panda finger and finger-pad collision geoms versus target bowl for grasping"]
        if task_key == "bowl_plate":
            allowances += ["target bowl versus main table and target plate support-contact pairs"]
        else:
            allowances += [
                "target bowl versus main table support-contact pairs",
                "named Panda finger and finger-pad collision geoms versus the top-drawer handle only",
                "target bowl versus top-drawer shell only for contact points within top_region plus 0.002 m tolerance",
            ]
        return {"sampling": "every actual physics substep and final control endpoint",
                "task_key": task_key,
                "monitored_scope": [
                    "robot links and gripper versus scene geometry except the task's explicit intended contact pairs",
                    "target bowl versus distractors, fixtures and non-finger robot geometry except task supports",
                    "MuJoCo collision geoms with contact distance <= 0 at every physics substep and final endpoint"],
                "allowances": allowances,
                "excluded_scope": list(cls.excluded_scope)}

    def __init__(self, env, *, task_key, stop_on_collision=True):
        description = self.rule_description(task_key)
        self.task_key = task_key
        self.required_objects = frozenset({"akita_black_bowl_1", "plate_1"} if task_key == "bowl_plate"
                                          else {"akita_black_bowl_1"})
        self.required_roles = frozenset({"target_bowl", "table", "robot", "robot_finger"}
                                        | ({"plate"} if task_key == "bowl_plate" else set()))
        self.monitored_scope, self.allowances = description["monitored_scope"], description["allowances"]
        self._drawer_region_site_id = None
        super().__init__(env, stop_on_collision=stop_on_collision)

    def _map_geometry(self):
        result = super()._map_geometry()
        if self.task_key == "bowl_plate":
            return result
        for record in result.values():
            if record["role"] == "plate":
                record["role"] = "scene"  # The plate is not this task's placement target.
        cabinet = self.core.fixtures_dict.get("wooden_cabinet_1")
        if cabinet is None:
            raise ValueError("drawer_bowl requires the registered wooden cabinet fixture")
        top = cabinet.get_obj().find(".//body[@name='wooden_cabinet_1_cabinet_top']")
        if top is None or top.find("joint[@name='wooden_cabinet_1_top_level']") is None:
            raise ValueError("Cannot verify top-drawer body and joint")
        site = top.find("site[@name='wooden_cabinet_1_top_region']")
        if site is None or site.get("type") != "box":
            raise ValueError("Cannot verify the top-drawer containment region")
        model = self.sim.model
        site_id = int(model.site_name2id(site.get("name")))
        top_body_id = int(model.body_name2id(top.get("name")))
        if int(model.site_bodyid[site_id]) != top_body_id:
            raise ValueError("Top-drawer region does not belong to the articulated top drawer")
        import numpy as np
        size = np.asarray(model.site_size[site_id], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Top-drawer containment region must be a finite positive box")
        by_name = {record["name"]: record for record in result.values()}
        grouped = {"drawer_shell": [], "drawer_handle": []}
        group = None
        for geom in top.findall("geom"):
            mesh = geom.get("mesh", "")
            if geom.get("type") == "mesh":
                if mesh.endswith("wooden_cabinet_top_handle_vis"):
                    group = "drawer_handle"
                elif mesh.endswith("wooden_cabinet_top_vis"):
                    group = "drawer_shell"
                else:
                    group = None
            record = by_name.get(geom.get("name"))
            if record is None:
                continue  # Non-collision visual mesh.
            if group is None or geom.get("type") != "box" or record["body"] != top.get("name"):
                raise ValueError("Unrecognized top-drawer collision geometry; cannot apply intended-contact allowance")
            record["role"] = group
            grouped[group].append(record["name"])
        # Match the inspected supported asset contract; an asset revision must
        # be reviewed, rather than silently allowing arbitrary extra geometry.
        if len(grouped["drawer_shell"]) != 6 or len(grouped["drawer_handle"]) != 3:
            raise ValueError("Top-drawer shell/handle collision groups differ from the supported asset")
        actual_top = {r["name"] for r in result.values() if r["body"] == top.get("name")}
        if actual_top != set(grouped["drawer_shell"] + grouped["drawer_handle"]):
            raise ValueError("Top-drawer collision geometry includes an unclassified part")
        self._drawer_region_site_id = site_id
        self.drawer_geometry_groups = grouped
        return result

    def _contact_reason(self, first, second, contact):
        roles = {first["role"], second["role"]}
        if roles == {"robot_finger", "drawer_handle"}:
            return None
        if roles == {"target_bowl", "drawer_shell"}:
            import numpy as np
            site = self._drawer_region_site_id
            center = np.asarray(self.sim.data.site_xpos[site], dtype=float)
            rotation = np.asarray(self.sim.data.site_xmat[site], dtype=float).reshape(3, 3)
            position = np.asarray(contact.pos, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("Drawer support contact must have a finite position")
            local = rotation.T @ (position - center)
            inside = np.all(np.abs(local) <= np.asarray(self.sim.model.site_size[site]) + self.drawer_support_tolerance_m)
            return None if inside else "target_bowl_drawer_exterior_contact"
        return super()._contact_reason(first, second, contact)


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
        return self._observation(result["observation"])

    def step(self, action: Sequence[float]) -> tuple[PolicyObservation, bool]:
        values = _validate_action(action)
        if not self.status or self.status.get("terminated"):
            raise SimulatorError("reset is required before stepping an inactive episode")
        result = self._request({"op": "step", "action": values})
        self.status = result["status"]
        return self._observation(result["observation"]), bool(self.status["success"])

    def _observation(self, data) -> PolicyObservation:
        observation = PolicyObservation.from_dict(data)
        expected = JOINT_OBSERVATION_KEYS if self.config.include_arm_joints else OBSERVATION_KEYS
        if set(data) != expected:
            raise SimulatorError("simulator observation fields do not match include_arm_joints configuration")
        return observation

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
        if config.initial_state_override is not None:
            self.metadata.update(
                benchmark_variant="LIBERO task with custom initial layout (explicit checkout)",
                initial_state_override=validate_initial_state_override(config.initial_state_override))
        if config.task_bddl_override is not None:
            self.metadata.update(
                benchmark_variant="LIBERO task with custom scene and initial layout (explicit checkout)",
                task_bddl_override=validate_task_bddl_override(config.task_bddl_override))
        if config.collision_check_mode != "none":
            monitoring = {"sampling": "every actual physics substep and final control endpoint",
                    "monitored_scope": list(_BowlObstacleCollisionMonitor.monitored_scope),
                    "allowances": list(_BowlObstacleCollisionMonitor.allowances),
                    "excluded_scope": list(_BowlObstacleCollisionMonitor.excluded_scope)}
            if config.collision_check_mode == "task_manipulation_v1":
                key = "bowl_plate" if config.suite == "libero_spatial" else "drawer_bowl"
                monitoring = _TaskManipulationCollisionMonitor.rule_description(key)
            monitoring.update(_collision_policy_metadata(config.stop_on_collision))
            self.metadata.update(collision_check_mode=config.collision_check_mode, collision_monitoring=monitoring,
                                 **_collision_policy_metadata(config.stop_on_collision))
        if config.include_arm_joints:
            self.metadata["arm_joint_observation"] = {
                "enabled": True, "position_key": "observation/joint_position",
                "velocity_key": "observation/joint_velocity", "count": 7,
                "order": [f"Panda arm joint {index}" for index in range(1, 8)],
                "position_unit": "rad", "velocity_unit": "rad/s",
                "source": "measured robot0_joint_pos and robot0_joint_vel observables; gripper joints remain in observation/state",
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
        if self.config.collision_check_mode == "bowl_obstacles_v1" and task_id != 0:
            raise ValueError("bowl_obstacles_v1 is defined only for LIBERO-Spatial bowl task 0")
        collision_task = None
        if self.config.collision_check_mode == "task_manipulation_v1":
            collision_task = _collision_task_key(self.config.suite, task_id)
        self._collision_monitor = None
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
        selected_state = initial_states[episode_index]
        override_metadata = None
        if self.config.initial_state_override is not None:
            selected_state, override_metadata = _load_initial_state_override(self.config.initial_state_override)
            if (self.config.task_bddl_override is None
                    and selected_state.shape != np.asarray(initial_states[episode_index]).shape):
                raise ValueError("initial_state_override shape must match the selected official task state")
        original_bddl_path = Path(self.benchmark.get_task_bddl_file_path(task_id)).resolve()
        bddl_path = original_bddl_path
        bddl_metadata = custom_contract = None
        if self.config.task_bddl_override is not None:
            bddl_metadata = validate_task_bddl_override(self.config.task_bddl_override)
            bddl_path = Path(bddl_metadata["path"])
            custom_contract = validate_task_bddl_contract(original_bddl_path, bddl_path)
            _validate_official_bddl_contract(original_bddl_path, bddl_path, custom_contract)
            if self.config.collision_check_mode == "bowl_obstacles_v1":
                expected_targets = {"akita_black_bowl_1": "akita_black_bowl", "plate_1": "plate"}
                if custom_contract["target_assets"] != expected_targets:
                    raise ValueError("bowl_obstacles_v1 requires the original target bowl and plate")
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
        if bddl_metadata is not None:
            expected_shape = np.asarray(self.env.sim.get_state().flatten()).shape
            if selected_state.shape != expected_shape:
                raise ValueError("initial_state_override shape must match the custom BDDL simulator model")
        obs = self.env.set_init_state(selected_state)
        for _ in range(self.config.settle_steps):
            obs, _reward, _done, _info = self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        raw_success = bool(self.env.check_success())
        if self.config.collision_check_mode == "bowl_obstacles_v1":
            self._collision_monitor = _BowlObstacleCollisionMonitor(
                self.env, stop_on_collision=self.config.stop_on_collision)
        elif self.config.collision_check_mode == "task_manipulation_v1":
            self._collision_monitor = _TaskManipulationCollisionMonitor(
                self.env, task_key=collision_task, stop_on_collision=self.config.stop_on_collision)
        if self._collision_monitor is not None:
            self.env.sim.forward()
            self._collision_monitor.inspect(phase="initial_state")
            self._collision_monitor.attach()
        collision = self._collision_monitor.snapshot() if self._collision_monitor is not None else None
        collision_stop = bool(self.config.stop_on_collision and collision and collision["detected"])
        success = raw_success and not collision_stop
        self.status = {"task_id": task_id, "episode_index": episode_index, "steps": 0,
                       "success": success, "terminated": (success and self.config.stop_on_success) or collision_stop,
                       "horizon_reached": False,
                       "initial_state_file": str(init_path), "initial_state_sha256": _sha256(init_path),
                       "bddl_sha256": _sha256(bddl_path)}
        if collision is not None:
            self.status.update(libero_success=raw_success, collision=collision,
                               collision_free_success=raw_success and not collision["detected"])
        if override_metadata is not None:
            self.status.update(initial_state_source="custom_override",
                               initial_state_override_file=override_metadata["path"],
                               initial_state_override_sha256=override_metadata["sha256"])
        if bddl_metadata is not None:
            scene_provenance = {
                "original_bddl_file": str(original_bddl_path), "original_bddl_sha256": _sha256(original_bddl_path),
                "task_bddl_override_file": bddl_metadata["path"], "task_bddl_override_sha256": bddl_metadata["sha256"],
                "custom_task_contract": custom_contract,
            }
            self.status.update(scene_provenance)
            self.metadata.update(scene_provenance)
        return self._response(obs)

    def step(self, action: Sequence[float]) -> dict:
        if self.env is None or self.status.get("terminated"):
            raise RuntimeError("reset is required before stepping")
        values = _validate_action(action)
        monitor = getattr(self, "_collision_monitor", None)
        if monitor is not None:
            monitor.begin_action(self.status["steps"])
        try:
            obs, _reward, _done, _info = self.env.step(values)
        except _ForbiddenCollisionStop:
            if monitor is None or monitor.first_event is None:
                raise
            # Finish the observation at the stopped physical state, without
            # executing the remainder of the control command or another step.
            obs = self.env.regenerate_obs_from_state(self.env.sim.get_state().flatten())
        if monitor is not None:
            monitor.finish_action()
        raw_success = bool(self.env.check_success())
        collision = monitor.snapshot() if monitor is not None else None
        collision_stop = bool(self.config.stop_on_collision and collision and collision["detected"])
        success = raw_success and not collision_stop
        self.status["steps"] += 1
        self.status["success"] = success
        self.status["horizon_reached"] = (self.config.max_steps is not None
                                          and self.status["steps"] >= self.config.max_steps)
        self.status["terminated"] = ((success and self.config.stop_on_success) or self.status["horizon_reached"]
                                     or collision_stop)
        if collision is not None:
            self.status.update(libero_success=raw_success, collision=collision,
                               collision_free_success=raw_success and not collision["detected"])
        return self._response(obs)

    def _response(self, obs: dict) -> dict:
        observation = PolicyObservation.from_libero(obs, self.prompt, image_size=self.config.image_size,
                                                     include_arm_joints=self.config.include_arm_joints)
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
