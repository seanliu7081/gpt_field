"""agent_SpeedUp_v2 generic, record-only contact monitor (evaluator side; spec §6.4).

Promoted from the Phase A prototype ``survey/infra/generic_contact_monitor.py``. It runs inside the simulator
worker process; nothing here reaches the policy observation, and a recorded contact never changes LIBERO
success or terminates an episode.

Roles come only from LIBERO's own registries (``objects_dict``, ``fixtures_dict``,
``robots[*].gripper.important_geoms``) and the MuJoCo body/joint tree; there is no per-task rule table and no
name list. Which object is "manipulated" is decided online from finger contacts.

Contact classes:

  robot_internal                       robot vs robot (ignored)
  finger_object                        finger/pad geom vs a movable object (intended grasp contact; allowed)
  finger_articulated_part              finger/pad vs a jointed fixture part (handle, door, knob; allowed).
                                       Deliberately looser than v1: ANY jointed part counts, goal or not
                                       (spec §6.4 allows the class). The drawer 003 replay's 99/101 gap is
                                       exactly this: 2 steps (366, 367) of finger contact on a non-goal
                                       drawer, forbidden for v1's task table, allowed here by design.
  robot_scene                          any robot geom vs arena / static fixture geometry, or non-finger robot
                                       geometry vs a jointed fixture part                (FORBIDDEN, recorded)
  robot_nonfinger_manipulated_object   palm/link geometry vs the manipulated object (finger-touched within
                                       the memory window, or a goal object)               (FORBIDDEN, recorded)
  robot_object                         any robot geom vs any other movable object
                                       (knocking a distractor)                            (FORBIDDEN, recorded)
  manipulated_object_support           manipulated object vs non-robot geometry, near-vertical contact normal
                                       (resting / being placed; allowed)
  manipulated_object_scene             manipulated object vs non-robot geometry, non-vertical normal
                                       (side or rim impact while carried)                 (FORBIDDEN, recorded)
  passive_scene                        contacts between untouched scene geometry (ignored)

"Manipulated" is a short memory, not a history: an object is manipulated while a finger touched it within the
last MANIPULATED_MEMORY_SUBSTEPS physics substeps (about 2 control steps). After a release the object's contacts
fall back to passive_scene once that window has passed, so a released object settling is not recorded.

An optional evaluator-only goal-aware tier is derived from the official BDDL goal atoms (no hand table): it
records support on a non-goal movable object, finger contact on a non-goal object (which then never becomes
"manipulated"), and support on a goal region's host outside that region. Covered goal regions: every ``In``
region (3-D site box) and every ``On`` region whose host is a registered fixture or object (the site's xy
footprint only, since an On region's site is a thin slab whose z band need not contain the support surface).
``On`` table zones are NOT covered: their host is the arena, and a manipulated object resting on the table
outside the zone (every pick from the table, every push_to) would be recorded as forbidden.

A contact counts when it is active: distance <= the contact's includemargin (= margin - gap), as
``agent_speedup_v2_truth.Scene.contacts`` does. Most LIBERO geoms have margin 0, so this is distance <= 0, as in
``gpt_field.libero._BowlObstacleCollisionMonitor``; a margin geom (the microwave's collision class, 1 mm) is
active at a small positive distance (counted in ``active_positive_distance_contacts``). Samples are taken after
every physics substep and at the control endpoint.

Non-perturbation: the endpoint refresh (``sim.forward``) restores ``qacc_warmstart`` afterwards
(``forward_keeping_warmstart``), so a monitored run follows the same physics as an unmonitored one. Before this
(contacts v1) the extra forward changed the solver's warm start, so pre-fix v2 traces with contacts replay
bitwise only under their own frozen module. The constants below are evaluator-only design defaults; they affect
diagnostics only (CONSTANT_LABELS).
"""

from __future__ import annotations

import copy
import math

import numpy as np

CONTACTS_VERSION = "agent-speedup-v2-generic-contact-v2"
MODE = "generic_contact_v2"
FORBIDDEN = ("robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene")
ALLOWED = ("finger_object", "finger_articulated_part", "manipulated_object_support")
IGNORED = ("robot_internal", "passive_scene")
SUPPORT_NORMAL_MAX_DEG = 35.0           # normal within 35 deg of vertical = support
SUPPORT_COS = math.cos(math.radians(SUPPORT_NORMAL_MAX_DEG))
MANIPULATED_MEMORY_SUBSTEPS = 50        # ~2 control steps at 25 physics substeps per step
REGION_TOLERANCE_M = 0.002              # goal-region box tolerance for the goal-aware tier
GOAL_REGION_PREDICATES = ("in", "on")
CONSTANT_LABELS = {
    "SUPPORT_NORMAL_MAX_DEG": "evaluator-only design default (diagnosis; never a control threshold)",
    "MANIPULATED_MEMORY_SUBSTEPS": "evaluator-only design default (diagnosis; ~2 control steps)",
    "REGION_TOLERANCE_M": "evaluator-only design default (diagnosis)",
}


def goal_objects_and_regions(core):
    """Movable goal objects and goal regions from the parsed BDDL goal.

    Returns (objects, regions, predicates): regions maps a goal site to its hosting registry owner (a fixture or
    object name, as the geometry map's ``owner``); predicates maps the site to "in" or "on". An On region
    without a registered host (a table zone, hosted by the arena) is not a goal region of this tier."""
    objects, regions, predicates = set(), {}, {}
    registered = set(core.objects_dict) | set(getattr(core, "fixtures_dict", {}) or {})
    for atom in core.parsed_problem["goal_state"]:
        predicate = str(atom[0]).lower()
        for arg in atom[1:]:
            arg = str(arg)
            if arg in core.objects_dict:
                objects.add(arg)
            elif arg in core.object_sites_dict and predicate in GOAL_REGION_PREDICATES:
                parent = getattr(core.object_sites_dict[arg], "parent_name", None)
                if predicate == "on" and parent not in registered:
                    continue                 # table zone: the host is the arena (see the module docstring)
                regions[arg] = parent
                predicates[arg] = predicate
                if parent in core.objects_dict:
                    objects.add(parent)      # a movable container (basket) hosting a goal region
    return objects, regions, predicates


def forward_keeping_warmstart(sim):
    """``sim.forward()`` without perturbing the next step: MuJoCo's forward pass rewrites ``qacc_warmstart``
    (the constraint solver's initial guess), so an extra evaluator-side forward would change the physics of a
    monitored run at solver tolerance. The warm start is restored afterwards; everything else a forward
    computes is recomputed by the next step."""
    data = getattr(sim.data, "_data", sim.data)
    warmstart = np.array(data.qacc_warmstart, copy=True)
    sim.forward()
    data.qacc_warmstart[:] = warmstart


def _box_local(data, site_id, points):
    center = np.asarray(data.site_xpos[site_id], dtype=float)
    rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    return (np.asarray(points, dtype=float) - center) @ rotation


class GenericContactMonitor:
    """Record-only contact classifier hooked on ``env.sim.step`` (one instance per episode)."""

    mode = MODE

    def __init__(self, env, *, goal_aware=True):
        self.env = env
        self.sim = env.sim
        self.core = getattr(env, "env", env)
        self.goal_aware = bool(goal_aware)
        self.goal_objects, self.goal_regions, self.goal_region_predicates = (
            goal_objects_and_regions(self.core) if self.goal_aware else (set(), {}, {}))
        self.first_event = None
        self.first_by_class = {}
        self.pairs_seen = {}
        self.counts = {"checks": 0, "physics_substeps": 0, "positive_distance_contacts_ignored": 0,
                       "active_positive_distance_contacts": 0,
                       **{f"points_{name}": 0 for name in (*FORBIDDEN, *ALLOWED, *IGNORED)},
                       "unmapped_contact_points": 0}
        self.touched = {}            # object -> last substep clock at which a finger touched it
        self.held_ever = set()
        self._substep_clock = 0
        self._action_index = None
        self._substep = 0
        self._original_step = self.sim.step
        self._attached = False
        self.geometry = self._map_geometry()

    # geometry ----------------------------------------------------------------------------------------------
    def _map_geometry(self):
        model = self.sim.model
        roots = {}
        for name, obj in self.core.objects_dict.items():
            roots[int(model.body_name2id(obj.root_body))] = (name, "object")
        for name, obj in self.core.fixtures_dict.items():
            roots[int(model.body_name2id(obj.root_body))] = (name, "fixture")
        finger_side = {}
        for index, robot in enumerate(self.core.robots):
            roots[int(model.body_name2id(robot.robot_model.root_body))] = (f"robot{index}", "robot")
            gripper = robot.gripper
            roots[int(model.body_name2id(gripper.root_body))] = (f"robot{index}_gripper", "robot")
            for key in ("left_finger", "left_fingerpad", "right_finger", "right_fingerpad"):
                for geom in gripper.important_geoms.get(key, []):
                    finger_side[geom] = key.split("_")[0]
        result = {}
        for geom_id in range(model.ngeom):
            if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                continue
            name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
            body_id = int(model.geom_bodyid[geom_id])
            current, owner, kind, jointed = body_id, "arena", "arena", False
            while True:
                if current in roots:
                    owner, kind = roots[current]
                    break
                if int(model.body_jntnum[current]) > 0:
                    jointed = True          # a joint between this body and its registered root
                if current == 0:
                    break
                current = int(model.body_parentid[current])
            role = kind
            if kind == "robot" and name in finger_side:
                role = "robot_finger"
            elif kind == "fixture":
                role = "fixture_part" if jointed else "fixture_static"
            result[geom_id] = {"id": geom_id, "name": name, "body": model.body_id2name(body_id) or f"body_{body_id}",
                               "owner": owner, "role": role, "finger_side": finger_side.get(name)}
        return result

    def role_summary(self):
        summary = {}
        for record in self.geometry.values():
            summary.setdefault(record["role"], set()).add(record["owner"])
        return {role: sorted(owners) for role, owners in sorted(summary.items())}

    # classification ----------------------------------------------------------------------------------------
    def _manipulated(self, owner):
        last = self.touched.get(owner)
        return last is not None and self._substep_clock - last <= MANIPULATED_MEMORY_SUBSTEPS

    @staticmethod
    def _normal_into(contact, object_is_geom2):
        normal = np.asarray(contact.frame[:3], dtype=float)  # MuJoCo: from geom1 towards geom2
        return normal if object_is_geom2 else -normal

    def classify(self, first, second, contact):
        robot = ("robot", "robot_finger")
        r1, r2 = first["role"], second["role"]
        if r1 in robot and r2 in robot:
            return "robot_internal"
        if r1 in robot or r2 in robot:
            bot, other = (first, second) if r1 in robot else (second, first)
            finger = bot["role"] == "robot_finger"
            if other["role"] == "object":
                if finger:
                    if self.goal_aware and self.goal_objects and other["owner"] not in self.goal_objects:
                        return "robot_object"        # finger on a non-goal (distractor) object
                    return "finger_object"
                return ("robot_nonfinger_manipulated_object"
                        if self._manipulated(other["owner"]) or other["owner"] in self.goal_objects
                        else "robot_object")
            if other["role"] == "fixture_part" and finger:
                return "finger_articulated_part"
            return "robot_scene"
        for obj, partner, is_geom2 in ((first, second, False), (second, first, True)):
            if obj["role"] == "object" and self._manipulated(obj["owner"]) and partner["owner"] != obj["owner"]:
                up = self._normal_into(contact, is_geom2)[2]
                if up < SUPPORT_COS:
                    return "manipulated_object_scene"
                if self.goal_aware:
                    hosted = [r for r, parent in self.goal_regions.items() if parent == partner["owner"]]
                    if hosted:
                        if not any(self._in_region(r, contact.pos) for r in hosted):
                            return "manipulated_object_scene"  # on the region host but outside the region (rim)
                    elif partner["role"] == "object" and partner["owner"] not in self.goal_objects:
                        return "manipulated_object_scene"      # resting on a distractor object
                return "manipulated_object_support"
        return "passive_scene"

    def _in_region(self, region, point):
        """In regions: inside the 3-D site box; On regions: inside the site's xy footprint (site frame)."""
        model, data = self.sim.model, self.sim.data
        site = int(model.site_name2id(region))
        local = np.abs(_box_local(data, site, [point])[0])
        limit = np.asarray(model.site_size[site], dtype=float) + REGION_TOLERANCE_M
        if self.goal_region_predicates.get(region) == "on":
            return bool(np.all(local[:2] <= limit[:2]))
        return bool(np.all(local <= limit))

    def _counts_as_manipulated(self, owner):
        """A finger contact makes an object manipulated, except a non-goal object under the goal-aware tier
        (that contact is recorded as robot_object and the distractor stays passive)."""
        return not (self.goal_aware and self.goal_objects and owner not in self.goal_objects)

    def inspect(self, *, phase, action_index=None, physics_substep=None):
        self.counts["checks"] += 1
        contacts, finger_contacts = [], {}
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            distance = float(contact.dist)
            if not math.isfinite(distance):
                raise ValueError("Non-finite simulator contact distance")
            margin = float(contact.includemargin)
            if distance > margin:
                self.counts["positive_distance_contacts_ignored"] += 1
                continue
            if distance > 0:
                self.counts["active_positive_distance_contacts"] += 1
            first, second = self.geometry.get(int(contact.geom1)), self.geometry.get(int(contact.geom2))
            if first is None or second is None:
                self.counts["unmapped_contact_points"] += 1
                continue
            contacts.append((first, second, contact, distance))
            for a, b in ((first, second), (second, first)):
                if a["role"] == "robot_finger" and b["role"] == "object":
                    finger_contacts.setdefault(b["owner"], set()).add(a["finger_side"])
        # A finger contact makes the object manipulated before this sample is classified.
        for owner, sides in finger_contacts.items():
            if self._counts_as_manipulated(owner):
                self.touched[owner] = self._substep_clock
            if {"left", "right"} <= sides:
                self.held_ever.add(owner)
        for first, second, contact, distance in contacts:
            kind = self.classify(first, second, contact)
            self.counts[f"points_{kind}"] += 1
            if kind in IGNORED:
                continue
            pair = tuple(sorted((first["owner"] + ":" + first["name"], second["owner"] + ":" + second["name"])))
            key = (kind, pair)
            if key not in self.pairs_seen:
                self.pairs_seen[key] = {"class": kind, "pair": list(pair), "first_action_index": action_index,
                                        "first_phase": phase, "points": 0}
            self.pairs_seen[key]["points"] += 1
            if kind not in self.first_by_class:
                self.first_by_class[kind] = {
                    "phase": phase, "action_index": action_index, "physics_substep": physics_substep,
                    "contact_distance_m": distance, "contact_pos": [float(v) for v in contact.pos],
                    "geom1": copy.deepcopy(first), "geom2": copy.deepcopy(second), "class": kind}
            if kind in FORBIDDEN and self.first_event is None:
                self.first_event = copy.deepcopy(self.first_by_class[kind])

    # step hook (same protocol as gpt_field.libero._BowlObstacleCollisionMonitor) ---------------------------
    def attach(self):
        if self._attached:
            raise RuntimeError("monitor is already attached")
        self.sim.step = self._physics_step
        self._attached = True

    def begin_action(self, action_index):
        self._action_index, self._substep = action_index, 0

    def _physics_step(self, *args, **kwargs):
        if self._action_index is None:
            raise RuntimeError("Monitored physics step is outside an acknowledged action")
        result = self._original_step(*args, **kwargs)
        self.counts["physics_substeps"] += 1
        self._substep_clock += 1
        current, self._substep = self._substep, self._substep + 1
        self.inspect(phase="action", action_index=self._action_index, physics_substep=current)
        return result

    def finish_action(self, *, refresh=True):
        """Endpoint sample. ``refresh=False`` when a chained monitor already ran ``sim.forward()``."""
        if self._substep == 0:
            raise RuntimeError("Collision monitor did not observe any actual physics substeps")
        if refresh:
            forward_keeping_warmstart(self.sim)
        self.inspect(phase="action", action_index=self._action_index, physics_substep=self._substep - 1)
        self._action_index = None

    def forbidden_points(self):
        return sum(self.counts[f"points_{name}"] for name in FORBIDDEN)

    def snapshot(self):
        by_class = {}
        for record in self.pairs_seen.values():
            by_class.setdefault(record["class"], []).append(record)
        return {"enabled": True, "mode": self.mode, "version": CONTACTS_VERSION,
                "detected": self.first_event is not None,
                "stop_on_collision": False, "collision_policy": "record_only",
                "success_qualification": "raw_libero_goal",
                "first_event": copy.deepcopy(self.first_event),
                "first_by_class": copy.deepcopy(self.first_by_class),
                "pairs_by_class": {k: sorted(v, key=lambda r: -r["points"])[:12] for k, v in sorted(by_class.items())},
                "manipulated_objects": sorted(self.touched), "held_objects": sorted(self.held_ever),
                "goal_aware": self.goal_aware, "goal_objects": sorted(self.goal_objects),
                "goal_regions": dict(self.goal_regions),
                "goal_region_predicates": dict(self.goal_region_predicates),
                "counts": {**self.counts, "forbidden_contact_points": self.forbidden_points()},
                "forbidden_classes": list(FORBIDDEN), "allowed_classes": list(ALLOWED),
                "ignored_classes": list(IGNORED)}
