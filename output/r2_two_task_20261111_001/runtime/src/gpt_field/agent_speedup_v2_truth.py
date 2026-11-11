"""agent_SpeedUp_v2 diagnosis-only strict physical truth (spec §6.5, AC-2).

Promoted from the Phase A prototypes ``survey/grounding/ground_lib.py`` and ``truth_lib.py``. DIAGNOSIS ONLY:
everything here reads privileged MuJoCo state (body poses, collision geoms, contacts, joint positions). It
runs in the simulator worker (finalization ops ``truth`` / ``settle_truth`` / ``answer_key``) and in
``scripts/agent_speedup_v2_truth_check.py``. It never feeds the policy and never changes the episode outcome.

Task-agnostic: predicates and arguments come from the env's parsed BDDL goal; container walls, supports and
rims are found from the collision geoms of the named objects, never from per-task constants. The only
constants are the generic physical tolerances in ``TOL``, identical for every task and not tuned to any run.

``explain(scene, atom, window)`` returns, per goal atom: ``raw`` (LIBERO's own ``_eval_predicate``), ``libero``
(the internals LIBERO tested), ``strict`` (the strict physical verdict), ``checks`` (each strict sub-check) and
``why_not`` (the failing sub-checks). Strict kinds: ``inside_container``, ``on_object``, ``on_fixture_region``,
``on_table_zone``, ``articulation``.

Orientation (decided 2026-10-18, lead default; the user may override; spec §6.5 D1). The ``on`` kinds
(``on_object``, ``on_fixture_region``, ``on_table_zone``) also require ``orientation_kept``: the object's body z
axis is within TOL["orientation_change_deg"] of its z axis at the episode's reset (recorded when the ``Scene`` is
built at reset), so a pot on its side or upside down on the burner, or a bowl upside down on the plate, is not a
strict success. ``inside`` does NOT require it: an object resting in the cavity with floor (or contents) support
counts even when it leans on a wall or the rim (a can lying in a basket, a bowl leaning on the drawer's front
wall), as long as it is not perched. ``not_perched_on_rim`` therefore asks for a SUPPORT contact (below the COM,
near-vertical normal) from the cavity or its contents at least TOL["rim_contact_m"] below the rim; the run-003
corner perch (every contact at rim height, no floor contact) stays lenient_success. The evidence always reports
the tilt from reset, whether orientation was required, and the share of the container normal force carried at
rim height. STRICT_ORIENTATION_KINDS is the one switch. The orientation check is deferrable like the rest
checks (a held object's orientation is the hand's). A Scene built later (``at_reset=False``) has no reset
orientations and the check fails closed.

Support contacts (all kinds) must lie below the COM AND have a contact normal within
TOL["support_normal_max_deg"] of vertical (the contact monitor's support rule). A container cavity is found
by rays along the region site's own x / y axes (a yawed container gets its own box, not a world-axis box).
On a fixture region, contacts with the fixture's jointed parts that do not carry the region (a stove knob)
are not support.

The module imports only numpy at import time (``mujoco`` lazily), so the pure helpers and the verdict logic are
testable in the project venv.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

TRUTH_VERSION = "agent-speedup-v2-truth-v3"

# Generic diagnosis tolerances (same for every task; NOT tuned to any run). Evaluator-only design defaults.
TOL = {
    "rest_lin_mps": 0.01,       # at rest: linear speed of the object body below 1 cm/s
    "rest_ang_rps": 0.10,       # ... and angular speed below 0.1 rad/s
    "rest_joint": 0.005,        # articulated joint speed below 5 mm/s or 5 mrad/s
    "rest_window_disp_mm": 1.0,  # windowed rest: origin moved < 1 mm over the last k control steps ...
    "rest_window_rot_deg": 1.0,  # ... and rotated < 1 degree
    "rest_joint_window": 0.001,  # windowed joint rest: q range < 1 mm / 1 mrad over the window
    "rim_below_m": 0.010,       # inside: object's lowest collision point >= 1 cm below the container rim
    "rim_contact_m": 0.005,     # inside: >= 1 support contact >= 5 mm below the rim (not perched on wall tops)
    "footprint_tol_m": 0.005,   # inside: below-rim vertices within the inner walls + 5 mm
    "zone_lift_m": 0.010,       # on-zone: lowest point within 1 cm of the support top (resting, not hovering)
    "below_com_m": 0.005,       # a support contact lies >= 5 mm below the object's COM ...
    "support_normal_max_deg": 35.0,  # ... and its normal is within 35 deg of vertical (as the contact monitor)
    "orientation_change_deg": 30.0,  # placed objects: body z axis within 30 deg of its reset z axis
}
# D1 (decided 2026-10-18, lead default; the user may override): 'on' requires orientation_kept, 'inside' does not.
STRICT_ORIENTATION_KINDS = ("on_object", "on_fixture_region", "on_table_zone")
WINDOW_STEPS = 10
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
STRICT_OK_KINDS = ("inside_container", "on_object", "on_fixture_region", "on_table_zone", "articulation")
# Sub-checks that fail only because the robot still holds or touches the object at the terminal state. AC-2:
# when every failing terminal sub-check is one of these (and a hold check is among them), the verdict is taken
# after the release-and-settle counterfactual. A held object is not at rest, so the rest checks belong here.
HELD_CHECKS = ("not_held", "robot_not_touching_part", "at_rest", "joint_at_rest", "orientation_kept")
HOLD_CHECKS = ("not_held", "robot_not_touching_part")
# The runner's release-and-settle counterfactual (spec §6.5): open 10, lift 8, settle 20 control steps.
RELEASE_SETTLE = {"open_steps": 10, "lift_steps": 8, "settle_steps": 20, "lift_action_z": 0.5}
CONSTANT_LABELS = {
    "TOL": "evaluator-only design defaults (diagnosis; identical for every task)",
    "STRICT_ORIENTATION_KINDS": "evaluator-only design default (which strict kinds require orientation_kept): "
                                "the 'on' kinds, not 'inside' (D1, decided 2026-10-18, lead default; the user "
                                "may override)",
    "WINDOW_STEPS": "evaluator-only design default (rest and contact window, control steps)",
    "RELEASE_SETTLE": "evaluator-only counterfactual protocol (spec §6.5)",
}


def _mujoco():
    import mujoco
    return mujoco


# ---------------- pure helpers ----------------
def r(x, n=4):
    """Round for JSON (lists, numpy arrays, bools, ints, floats); None stays None."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if isinstance(x, np.ndarray):
        return [r(v, n) for v in x.tolist()]
    if isinstance(x, (int, np.integer)):
        return int(x)
    value = float(x)
    return round(value, n) if math.isfinite(value) else None


def jsonable(value):
    """Recursively convert numpy scalars/arrays, tuples and sets; non-finite floats become None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def hull2d(pts):
    pts = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def hull_margin(hull, q):
    """Signed distance of q to the hull boundary (+ inside). Degenerate hulls give -distance to the segment."""
    q = np.asarray(q, float)
    if len(hull) == 0:
        return None
    if len(hull) < 3:
        a, b = np.asarray(hull[0], float), np.asarray(hull[-1], float)
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(q - a, ab) / np.dot(ab, ab), 0, 1))
        return -float(np.linalg.norm(q - (a + t * ab)))
    inside, best = True, float("inf")
    n = len(hull)
    for i in range(n):
        a, b = np.asarray(hull[i]), np.asarray(hull[(i + 1) % n])
        e = b - a
        if e[0] * (q[1] - a[1]) - e[1] * (q[0] - a[0]) < 0:
            inside = False
        t = float(np.clip(np.dot(q - a, e) / np.dot(e, e), 0, 1))
        best = min(best, float(np.linalg.norm(q - (a + t * e))))
    return best if inside else -best


# ---------------- evaluator scene ----------------
class Scene:
    """Thin evaluator view of a LIBERO OffScreenRenderEnv (``env``) and its BDDL problem (``env.env``).

    Build it at the episode's reset (``at_reset=True``): it then records every movable object's body z axis,
    the reference of the ``orientation_kept`` check. ``at_reset=False`` records none (the check fails)."""

    def __init__(self, env, *, at_reset=True):
        mj = _mujoco()
        self.mj = mj
        self.env = env
        self.core = env.env
        self.sim = env.sim
        self.M = env.sim.model._model
        self.D = env.sim.data._data
        M = self.M
        self.parent = np.asarray(M.body_parentid, int)
        self.geom_body = np.asarray(M.geom_bodyid, int)
        self.coll = (np.asarray(M.geom_contype) != 0) | (np.asarray(M.geom_conaffinity) != 0)
        self.inv_root = {int(b): n for n, b in self.core.obj_body_id.items()}
        self._owner_cache = {}
        self._subtree_cache = {}
        self.robot_bodies = {b for b in range(M.nbody) if self.bname(b).startswith(ROBOT_PREFIXES)}
        self.robot_geoms = {g for g in range(M.ngeom) if self.coll[g] and int(self.geom_body[g]) in self.robot_bodies}
        self.reset_zaxis = ({name: self.zaxis(self.core.obj_body_id[name]) for name in self.core.objects_dict}
                            if at_reset else None)

    # names / ownership
    def bname(self, b):
        return self.mj.mj_id2name(self.M, self.mj.mjtObj.mjOBJ_BODY, int(b)) or f"body_{b}"

    def gname(self, g):
        return self.mj.mj_id2name(self.M, self.mj.mjtObj.mjOBJ_GEOM, int(g)) or f"geom_{g}"

    def owner(self, b):
        """Nearest ancestor that is a BDDL object/fixture root; else the world-child body name."""
        b = int(b)
        if b in self._owner_cache:
            return self._owner_cache[b]
        c, top = b, b
        while c != 0:
            if c in self.inv_root:
                self._owner_cache[b] = self.inv_root[c]
                return self.inv_root[c]
            top = c
            c = int(self.parent[c])
        name = self.bname(top)
        if name.startswith(ROBOT_PREFIXES) or b in self.robot_bodies:
            name = "robot"
        self._owner_cache[b] = name
        return name

    def subtree(self, root):
        root = int(root)
        if root not in self._subtree_cache:
            out = set()
            for b in range(self.M.nbody):
                c = b
                while c not in (0, root):
                    c = int(self.parent[c])
                if c == root and (b == root or root != 0):
                    out.add(b)
            self._subtree_cache[root] = out
        return self._subtree_cache[root]

    def object_bodies(self, name):
        return self.subtree(self.core.obj_body_id[name])

    def geoms_of(self, bodies):
        return sorted(g for g in range(self.M.ngeom) if self.coll[g] and int(self.geom_body[g]) in bodies)

    # geometry
    def geom_pts(self, g):
        mj, M, D = self.mj, self.M, self.D
        t = int(M.geom_type[g])
        R = D.geom_xmat[g].reshape(3, 3)
        p = D.geom_xpos[g]
        s = M.geom_size[g]
        if t == mj.mjtGeom.mjGEOM_MESH:
            mid = int(M.geom_dataid[g])
            a, n = int(M.mesh_vertadr[mid]), int(M.mesh_vertnum[mid])
            v = np.asarray(M.mesh_vert[a:a + n], float)
        elif t == mj.mjtGeom.mjGEOM_BOX:
            v = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * s
        elif t in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            h = s[1]
            v = [[s[0] * math.cos(a), s[0] * math.sin(a), z] for a in ang for z in (-h, h)]
            if t == mj.mjtGeom.mjGEOM_CAPSULE:
                v += [[0, 0, h + s[0]], [0, 0, -h - s[0]]]
            v = np.asarray(v, float)
        elif t in (mj.mjtGeom.mjGEOM_SPHERE, mj.mjtGeom.mjGEOM_ELLIPSOID):
            radii = s if t == mj.mjtGeom.mjGEOM_ELLIPSOID else np.array([s[0]] * 3)
            v = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                          if (a, b, c) != (0, 0, 0)], float)
            v = v / np.linalg.norm(v, axis=1, keepdims=True) * radii
        else:  # plane / hfield: centre only
            v = np.zeros((1, 3))
        return v @ R.T + p

    def pts_of(self, geoms):
        return np.vstack([self.geom_pts(g) for g in geoms]) if geoms else np.zeros((0, 3))

    def aabb(self, g):
        p = self.geom_pts(g)
        return p.min(0), p.max(0)

    def com(self, name):
        return np.asarray(self.D.subtree_com[self.core.obj_body_id[name]], float).copy()

    def body_vel(self, b):
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.M, self.D, self.mj.mjtObj.mjOBJ_BODY, int(b), res, 0)
        return res[3:], res[:3]  # linear, angular (world frame)

    def tilt_deg(self, b):
        R = self.D.xmat[int(b)].reshape(3, 3)
        return float(math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))

    def zaxis(self, b):
        return np.asarray(self.D.xmat[int(b)], float).reshape(3, 3)[:, 2].copy()

    def tilt_change_deg(self, name):
        """Angle (deg) between the object's body z axis now and at reset; None without a reset reference."""
        reference = (self.reset_zaxis or {}).get(name)
        if reference is None:
            return None
        cosine = float(np.dot(self.zaxis(self.core.obj_body_id[name]), reference))
        return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))

    # contacts
    def contacts(self, geoms):
        """Active contacts of ``geoms`` against everything else: dist <= includemargin (= margin - gap).

        Most LIBERO geoms have margin 0 (active iff penetrating); the microwave's collision class uses a 1 mm
        margin, so objects resting on its floor sit at dist ~ +1 mm and must still count as support."""
        gs = set(geoms)
        out = []
        f6 = np.zeros(6)
        for i in range(int(self.D.ncon)):
            c = self.D.contact[i]
            if c.dist > c.includemargin:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in gs) == (g2 in gs):
                continue
            own, oth = (g1, g2) if g1 in gs else (g2, g1)
            self.mj.mj_contactForce(self.M, self.D, i, f6)
            n = np.asarray(c.frame[:3], float)
            out.append({"own": self.gname(own), "other": self.gname(oth), "other_g": oth,
                        "owner": self.owner(self.geom_body[oth]), "other_body": self.bname(self.geom_body[oth]),
                        "pos": np.asarray(c.pos, float).copy(), "fn": float(f6[0]), "nz": float(abs(n[2])),
                        "dist": float(c.dist)})
        return out

    def robot_contacts(self, geoms):
        return [c for c in self.contacts(geoms) if c["other_g"] in self.robot_geoms]

    # rays
    def ray(self, p, v, skip_bodies=(), maxd=1.0):
        """First collision geom along p + t v (t <= maxd), skipping visual geoms and ``skip_bodies``."""
        p = np.asarray(p, float).copy()
        v = np.asarray(v, float) / np.linalg.norm(v)
        gid = np.array([-1], dtype=np.int32)
        travelled = 0.0
        for _ in range(64):
            d = self.mj.mj_ray(self.M, self.D, p, v, None, 1, -1, gid)
            if d < 0 or travelled + d > maxd:
                return None
            g = int(gid[0])
            travelled += d
            if self.coll[g] and int(self.geom_body[g]) not in skip_bodies:
                return {"geom": self.gname(g), "g": g, "body": self.bname(self.geom_body[g]),
                        "owner": self.owner(self.geom_body[g]), "dist": travelled}
            p = p + v * (d + 1e-4)
            travelled += 1e-4
        return None

    # sites / regions
    def site(self, name):
        sid = self.mj.mj_name2id(self.M, self.mj.mjtObj.mjOBJ_SITE, name)
        so = self.core.object_sites_dict[name]
        return {"sid": sid, "body": int(self.M.site_bodyid[sid]),
                "pos": np.asarray(self.D.site_xpos[sid], float).copy(),
                "mat": np.asarray(self.D.site_xmat[sid], float).reshape(3, 3).copy(),
                "size": np.asarray(so.size, float), "obj": so,
                "parent": getattr(so, "parent_name", None), "joints": list(getattr(so, "joints", None) or [])}

    @staticmethod
    def libero_box(site):
        """Exactly SiteObject.in_box / TargetZone.in_box: centre +- |mat @ size|, bottom lowered 1 cm."""
        tot = np.abs(site["mat"] @ site["size"])
        ub, lb = site["pos"] + tot, site["pos"] - tot
        lb[2] -= 0.01
        return lb, ub

    @staticmethod
    def libero_under(site, p, other_height=0.10):
        """Exactly SiteObject.under (it uses mat @ d, not mat.T @ d, and the unrotated size)."""
        s = site["size"]
        d = site["mat"] @ (np.asarray(p) - site["pos"])
        return bool(s[2] - 0.005 < d[2] < s[2] + other_height and np.all(np.abs(d[:2]) < s[:2])), d

    def dynamic_bodies(self):
        out = set(self.robot_bodies)
        for n in self.core.objects_dict:
            out |= self.object_bodies(n)
        return out

    def cavity(self, site):
        """Container cavity around a region site, found by rays from the site centre (movables/robot skipped).

        floor = first geom straight down; ceiling = first geom straight up if it belongs to the site's own body
        subtree (closed container); walls = first geoms of that subtree hit by 8 horizontal rays at mid-height,
        cast at k x 45 deg in the SITE frame (0 deg = the site's x axis projected on the horizontal plane, 90 deg =
        its y axis). rim = ceiling for closed containers, else the LOWEST wall top. The cavity rectangle spans the
        axis-ray wall distances along the site's x / y axes (the site's half-size on open sides);
        ``cavity_xy_hull`` is that rectangle in world xy, ``cavity_xy_box`` its world [x-, x+, y-, y+] bounds."""
        cav = self.subtree(site["body"])
        cont = self.owner(site["body"])
        own = self.object_bodies(cont) if cont in self.core.obj_body_id else set()
        skip = self.dynamic_bodies() - cav - own
        lb, ub = self.libero_box(site)
        c = site["pos"]

        def incont(h):
            return h is not None and int(self.geom_body[h["g"]]) in cav
        dn = self.ray(c, [0, 0, -1], skip_bodies=skip, maxd=0.5)
        up = self.ray(c, [0, 0, 1], skip_bodies=skip, maxd=0.3)
        floor_z = None if dn is None else c[2] - dn["dist"]
        ceil_z = c[2] + up["dist"] if incont(up) else None
        top = min(ub[2], ceil_z) if ceil_z is not None else ub[2]
        mid = (top + (floor_z if floor_z is not None else lb[2] + 0.01)) / 2
        axes = []
        for column in (0, 1):
            flat = np.array([site["mat"][0, column], site["mat"][1, column], 0.0])
            norm = float(np.linalg.norm(flat))
            axes.append(flat / norm if norm > 1e-9 else np.eye(3)[column])
        walls, open_dirs, dists, ext = {}, [], {}, {}
        for k in range(8):
            a = k * math.pi / 4
            v = math.cos(a) * axes[0] + math.sin(a) * axes[1]
            v = v / np.linalg.norm(v)
            h = self.ray([c[0], c[1], mid], v, skip_bodies=skip, maxd=0.6)
            key = f"{k * 45}deg"
            if not incont(h):
                open_dirs.append(key)
                dists[key] = None if h is None else (round(h["dist"], 4), h["geom"], "outside_container")
                continue
            walls[h["g"]] = float(self.aabb(h["g"])[1][2])
            dists[key] = (round(h["dist"], 4), h["geom"])
            if k % 2 == 0:
                ext[k * 45] = h["dist"]
        size = np.asarray(site["size"], float)
        extents = [-ext.get(180, size[0]), ext.get(0, size[0]), -ext.get(270, size[1]), ext.get(90, size[1])]
        corners = [np.asarray(c[:2], float) + u * axes[0][:2] + w * axes[1][:2]
                   for u in extents[:2] for w in extents[2:]]
        hull = hull2d(corners)
        box = [min(p[0] for p in corners), max(p[0] for p in corners),
               min(p[1] for p in corners), max(p[1] for p in corners)]
        rim = (ceil_z if ceil_z is not None else min(walls.values())) if walls else ceil_z
        return {"floor_z": floor_z, "floor_geom": None if dn is None else dn["geom"], "floor_in_container": incont(dn),
                "ceiling_z": ceil_z, "ceiling_geom": up["geom"] if incont(up) else None,
                "mid_z": mid, "walls": {self.gname(g): z for g, z in walls.items()}, "rim_z": rim,
                "rim_kind": "ceiling" if ceil_z is not None else "lowest_wall_top",
                "rim_max_z": max(walls.values()) if walls else None, "open_dirs": open_dirs, "wall_dists": dists,
                "cavity_xy_box": box, "cavity_xy_hull": hull, "cavity_site_frame_extents": extents,
                "cavity_axes_xy": [axes[0][:2].tolist(), axes[1][:2].tolist()], "bodies": cav}

    def joint_info(self, jname):
        mj = self.mj
        jid = mj.mj_name2id(self.M, mj.mjtObj.mjOBJ_JOINT, jname)
        adr, dadr = int(self.M.jnt_qposadr[jid]), int(self.M.jnt_dofadr[jid])
        return {"joint": jname, "q": float(self.D.qpos[adr]), "qd": float(self.D.qvel[dadr]),
                "range": [float(x) for x in self.M.jnt_range[jid]], "type": int(self.M.jnt_type[jid]),
                "body": int(self.M.jnt_bodyid[jid])}

    def goal(self):
        return [[str(a) for a in atom] for atom in self.core.parsed_problem["goal_state"]]


class Window:
    """Rolling evidence for the last k control steps.

    Contacts flicker on soft/margin geoms and velocities chatter at ~1 cm/s with < 0.05 mm displacement, so
    instantaneous tests are unreliable. ``update()`` once per control step; ``explain`` then uses windowed rest
    (displacement / rotation over the window) and the union of contacts seen in the window."""

    def __init__(self, k=WINDOW_STEPS):
        self.k = k
        self.buf, self.jbuf, self._geoms = {}, {}, {}

    def update(self, scene, names, joints=()):
        for n in names:
            if n not in self._geoms:
                self._geoms[n] = scene.geoms_of(scene.object_bodies(n))
            b = scene.core.obj_body_id[n]
            rec = (np.asarray(scene.D.xpos[b], float).copy(), np.asarray(scene.D.xquat[b], float).copy(),
                   scene.contacts(self._geoms[n]))
            self.buf.setdefault(n, deque(maxlen=self.k)).append(rec)
        for j in joints:
            self.jbuf.setdefault(j, deque(maxlen=self.k)).append(scene.joint_info(j)["q"])

    def reset(self):
        self.buf.clear()
        self.jbuf.clear()


def joint_targets(scene, name):
    core = scene.core
    if name in core.object_sites_dict:
        return list(core.object_sites_dict[name].joints or [])
    return list(core.get_object(name).joints)


def window_targets(scene):
    """(goal objects, goal joints) whose evidence the window must keep."""
    goal = scene.goal()
    objs = sorted({a[1] for a in goal if a[1] in scene.core.objects_dict})
    joints = sorted({j for a in goal if a[0].lower() in ("open", "close", "turnon", "turnoff")
                     for j in joint_targets(scene, a[1])})
    return objs, joints


def _rest(scene, name):
    lin, ang = scene.body_vel(scene.core.obj_body_id[name])
    return (float(np.linalg.norm(lin)) < TOL["rest_lin_mps"] and float(np.linalg.norm(ang)) < TOL["rest_ang_rps"],
            float(np.linalg.norm(lin)), float(np.linalg.norm(ang)))


def _qang(a, b):
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def _obj(scene, name, window):
    geoms = scene.geoms_of(scene.object_bodies(name))
    pts = scene.pts_of(geoms)
    origin = np.asarray(scene.D.xpos[scene.core.obj_body_id[name]], float).copy()
    con = scene.contacts(geoms)
    rest, v, w = _rest(scene, name)
    rest_src, disp_mm, rot_deg = "instant_velocity", None, None
    hist = window.buf.get(name) if window is not None else None
    if hist is not None and len(hist) >= min(5, window.k):
        q_now = np.asarray(scene.D.xquat[scene.core.obj_body_id[name]], float)
        disp_mm = max(float(np.linalg.norm(p - origin)) for p, _, _ in hist) * 1000
        rot_deg = max(_qang(q, q_now) for _, q, _ in hist)
        rest = disp_mm < TOL["rest_window_disp_mm"] and rot_deg < TOL["rest_window_rot_deg"]
        rest_src = f"window_{len(hist)}_steps"
        seen = {}
        for _, _, cs in hist:
            for c in cs:
                seen[(c["own"], c["other"])] = c
        for c in con:
            seen[(c["own"], c["other"])] = c
        con = list(seen.values())
    robot = [c for c in con if c["other_g"] in scene.robot_geoms]
    env = [c for c in con if c["other_g"] not in scene.robot_geoms]
    return {"name": name, "geoms": geoms, "pts": pts, "origin": origin, "com": scene.com(name),
            "low": float(pts[:, 2].min()), "high": float(pts[:, 2].max()), "robot": robot, "env": env,
            "rest": rest, "v": v, "w": w, "rest_src": rest_src, "disp_mm": disp_mm, "rot_deg": rot_deg,
            "tilt": scene.tilt_deg(scene.core.obj_body_id[name]),
            "tilt_change": scene.tilt_change_deg(name)}


def _below_com(o, c):
    """A support contact: below the COM and with a near-vertical normal (not a side or rim contact)."""
    return (c["pos"][2] < o["com"][2] - TOL["below_com_m"]
            and c["nz"] >= math.cos(math.radians(TOL["support_normal_max_deg"])))


def _orientation_kept(o):
    return o["tilt_change"] is not None and o["tilt_change"] <= TOL["orientation_change_deg"]


def _with_orientation(checks, o, kind):
    if kind in STRICT_ORIENTATION_KINDS:
        checks["orientation_kept"] = _orientation_kept(o)
    return checks


def inside_checks(o, *, cavity_contacts, contents_contacts, outside_contacts, rim_z, rim_kind, footprint_margin,
                  com_margin):
    """Strict 'inside' sub-checks from measured evidence (pure; ``explain_in`` gathers the inputs).

    ``cavity_contacts``: the object's contacts with the container's cavity bodies; ``contents_contacts``: its
    contacts with other movable objects whose origin lies in the region box; ``outside_contacts``: every other
    environment contact. ``footprint_margin`` is the worst below-rim vertex margin to the inner walls (None when no
    vertex is below the rim) and ``com_margin`` the COM's margin to the cavity rectangle.

    D1: no orientation check; not perched = some support contact (below the COM, near-vertical normal) from the
    cavity or its contents lies >= TOL["rim_contact_m"] below the rim (or the cavity is closed by a ceiling). A
    lean on a wall or rim with floor support passes; a perch whose contacts are all at rim height does not.
    Returns (checks, supports, supports_below_rim)."""
    supports = [c for c in list(cavity_contacts) + list(contents_contacts) if _below_com(o, c)]
    below_rim = [c for c in supports if c["pos"][2] <= rim_z - TOL["rim_contact_m"]]
    checks = {
        "not_held": not o["robot"],
        "at_rest": o["rest"],
        "supported_by_container_or_contents": bool(supports),
        "no_outside_support": not [c for c in outside_contacts if _below_com(o, c)],
        "lowest_point_below_rim": o["low"] <= rim_z - TOL["rim_below_m"],
        "not_perched_on_rim": rim_kind == "ceiling" or bool(below_rim),
        "below_rim_footprint_inside_walls": footprint_margin is not None
                                            and footprint_margin >= -TOL["footprint_tol_m"],
        "com_over_cavity": com_margin is not None and com_margin >= 0,
    }
    _with_orientation(checks, o, "inside_container")     # a no-op under D1 (kept so the switch is the one place)
    return checks, supports, below_rim


def rim_height_force_share(cavity_contacts, rim_z):
    """Share of the container contacts' normal force carried at rim height (evidence only), None without force."""
    total = sum(max(0.0, c["fn"]) for c in cavity_contacts)
    rim = sum(max(0.0, c["fn"]) for c in cavity_contacts if c["pos"][2] > rim_z - TOL["rim_contact_m"])
    return rim / total if total > 0 else None


def _site_xy_hull(site):
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * site["size"]
    w = corners @ site["mat"].T + site["pos"]
    return hull2d(w[:, :2])


def _common_numbers(o):
    return {"origin": r(o["origin"]), "com": r(o["com"]), "low_z": r(o["low"]), "high_z": r(o["high"]),
            "origin_minus_low_mm": r((o["origin"][2] - o["low"]) * 1000, 2), "speed_mps": r(o["v"], 5),
            "ang_rps": r(o["w"], 4), "tilt_deg": r(o["tilt"], 2),
            "tilt_change_from_reset_deg": r(o["tilt_change"], 2), "orientation_kept": _orientation_kept(o),
            "rest_source": o["rest_src"],
            "window_disp_mm": r(o["disp_mm"], 3), "window_rot_deg": r(o["rot_deg"], 3),
            "robot_contacts": sorted({c["other"] for c in o["robot"]}),
            "env_contacts": sorted({f'{c["owner"]}:{c["other"]}' for c in o["env"]})}


def explain_in(scene, obj, region, window):
    core = scene.core
    site = scene.site(region)
    o = _obj(scene, obj, window)
    lb, ub = scene.libero_box(site)
    p = o["origin"]
    raw = bool(core._eval_predicate(["in", obj, region]))
    lib = {"tested_point": "object body ORIGIN only", "origin": r(p), "box_lb": r(lb), "box_ub": r(ub),
           "margin_to_ub_mm": r((ub - p) * 1000, 2), "margin_to_lb_mm": r((p - lb) * 1000, 2),
           "contact_test": "SiteObjectState.check_contact is always True"}
    cv = scene.cavity(site)
    cav_bodies = cv["bodies"]
    hull = _site_xy_hull(site)
    bx = cv["cavity_xy_box"]
    cav_hull = cv["cavity_xy_hull"]
    rim_z = cv["rim_z"] if cv["rim_z"] is not None else float(ub[2])
    cav_con = [c for c in o["env"] if int(scene.geom_body[c["other_g"]]) in cav_bodies]
    movable = set(core.objects_dict.keys()) - {obj}
    mov_con = [c for c in o["env"] if c["owner"] in movable and int(scene.geom_body[c["other_g"]]) not in cav_bodies]
    mov_in = []
    for c in mov_con:
        q = np.asarray(scene.D.xpos[core.obj_body_id[c["owner"]]])
        mov_in.append(bool(np.all(q > lb) and np.all(q < ub)))
    ids = {id(c) for c in cav_con} | {id(c) for c in mov_con}
    other_con = [c for c in o["env"] if id(c) not in ids]
    below = o["pts"][o["pts"][:, 2] < rim_z]
    fp = [hull_margin(cav_hull, v[:2]) for v in below] if len(below) else []
    fp_min = min(fp) if fp else None
    fp_reg = [hull_margin(hull, v[:2]) for v in below] if len(below) else []
    com_m = hull_margin(cav_hull, o["com"][:2])
    contents = [c for c, ok in zip(mov_con, mov_in) if ok]
    # Contents contacts whose owner is outside the region box are neither cavity nor contents support; they stay
    # in mov_con (ids) and so never count as outside support either (as before).
    checks, supports, supports_below_rim = inside_checks(
        o, cavity_contacts=cav_con, contents_contacts=contents, outside_contacts=other_con, rim_z=rim_z,
        rim_kind=cv["rim_kind"], footprint_margin=fp_min, com_margin=com_m)
    # Evidence only (D1): the share of the container normal force carried at rim height and the tilt from reset.
    rim_share = rim_height_force_share(cav_con, rim_z)
    nums = _common_numbers(o)
    nums.update({"container_part": scene.bname(site["body"]), "rim_kind": cv["rim_kind"], "rim_z": r(rim_z),
                 "region_top_z": r(ub[2]), "region_top_minus_rim_mm": r((ub[2] - rim_z) * 1000, 2),
                 "low_below_rim_mm": r((rim_z - o["low"]) * 1000, 2),
                 "low_above_floor_mm": r((o["low"] - cv["floor_z"]) * 1000, 2) if cv["floor_z"] is not None else None,
                 "walls_top_z": {k: r(v) for k, v in cv["walls"].items()}, "floor_z": r(cv["floor_z"]),
                 "floor_geom": cv["floor_geom"], "ceiling_z": r(cv["ceiling_z"]), "open_dirs": cv["open_dirs"],
                 "cavity_xy_box": r(bx), "region_xy_box": [r(lb[0]), r(ub[0]), r(lb[1]), r(ub[1])],
                 "container_contacts": [[c["other"], r(c["pos"]), r(c["fn"], 3)] for c in cav_con][:12],
                 "max_container_contact_z": r(max([c["pos"][2] for c in cav_con], default=None)),
                 "rim_height_normal_force_share": r(rim_share, 3),
                 "support_contacts": len(supports), "support_contacts_below_rim": len(supports_below_rim),
                 "deepest_support_contact_below_rim_mm": r(max([(rim_z - c["pos"][2]) * 1000
                                                                for c in supports], default=None), 2),
                 "cavity_site_frame_extents": r(cv["cavity_site_frame_extents"]),
                 "contents_contacts": sorted({c["owner"] for c in mov_con}),
                 "outside_contacts": sorted({f'{c["owner"]}:{c["other"]}' for c in other_con}),
                 "below_rim_vertices": int(len(below)),
                 "footprint_worst_margin_mm_walls": r(fp_min * 1000 if fp_min is not None else None, 2),
                 "footprint_worst_margin_mm_region": r(min(fp_reg) * 1000 if fp_reg else None, 2),
                 "com_margin_mm_walls": r(com_m * 1000 if com_m is not None else None, 2),
                 "obj_xy_extent": [r(o["pts"][:, 0].min()), r(o["pts"][:, 0].max()),
                                   r(o["pts"][:, 1].min()), r(o["pts"][:, 1].max())]})
    return raw, lib, checks, nums


def explain_on_object(scene, obj, target, window):
    core = scene.core
    o = _obj(scene, obj, window)
    raw = bool(core._eval_predicate(["on", obj, target]))
    pt = np.asarray(scene.D.xpos[core.obj_body_id[target]], float)
    contact = bool(core.check_contact(core.get_object(target), core.get_object(obj)))
    dxy = float(np.linalg.norm(o["origin"][:2] - pt[:2]))
    lib = {"tested": "target_origin_z <= obj_origin_z AND any contact AND |origin_xy - target_origin_xy| < 0.03",
           "z_ok": bool(pt[2] <= o["origin"][2]), "contact": contact, "xy_dist_mm": r(dxy * 1000, 2),
           "xy_ok": dxy < 0.03}
    tb = scene.object_bodies(target)
    th = hull2d(scene.pts_of(scene.geoms_of(tb))[:, :2])
    sup = [c for c in o["env"] if _below_com(o, c)]
    sup_t = [c for c in sup if int(scene.geom_body[c["other_g"]]) in tb]
    sup_other = [c for c in sup if int(scene.geom_body[c["other_g"]]) not in tb]
    com_m = hull_margin(th, o["com"][:2])
    sp = hull2d([c["pos"][:2] for c in sup_t]) if sup_t else []
    stab = hull_margin(sp, o["com"][:2]) if sp else None
    checks = {
        "not_held": not o["robot"],
        "at_rest": o["rest"],
        "supported_by_target": bool(sup_t),
        "supported_only_by_target": bool(sup_t) and not sup_other,
        "com_over_target_footprint": com_m is not None and com_m >= 0,
    }
    _with_orientation(checks, o, "on_object")
    nums = _common_numbers(o)
    nums.update({"target_supports": len(sup_t),
                 "other_supports": sorted({f'{c["owner"]}:{c["other"]}' for c in sup_other}),
                 "com_margin_target_mm": r(com_m * 1000 if com_m is not None else None, 2),
                 "com_margin_support_polygon_mm": r(stab * 1000 if stab is not None else None, 2)})
    return raw, lib, checks, nums


def explain_on_site(scene, obj, region, window):
    core = scene.core
    site = scene.site(region)
    o = _obj(scene, obj, window)
    raw = bool(core._eval_predicate(["on", obj, region]))
    under, d = scene.libero_under(site, o["origin"])
    parent = site["parent"]
    pobj = core.get_object(parent) if parent else None
    contact = bool(core.check_contact(pobj, core.get_object(obj))) if pobj is not None else None
    s = site["size"]
    lib = {"tested": "origin in site z-band [top-5mm, top+10cm] and |xy| < half-size (frame: mat @ d)"
                     + (" AND any contact with parent fixture" if pobj is not None else " (NO contact test: zone)"),
           "under": under, "d_site": r(d), "band_z": [r(s[2] - 0.005), r(s[2] + 0.10)], "half_xy": r(s[:2]),
           "parent": parent, "parent_contact": contact}
    hull = _site_xy_hull(site)
    excluded = set()
    if pobj is not None:
        sup_bodies = scene.object_bodies(parent)
        # Jointed parts of the host that do not carry the region (a stove knob) are not support: their subtrees
        # leave the support set, so contacts on them count as other support.
        for j in range(int(scene.M.njnt)):
            body = int(scene.M.jnt_bodyid[j])
            if body in sup_bodies:
                part = scene.subtree(body)
                if site["body"] not in part:
                    excluded |= part
        sup_bodies = sup_bodies - excluded
    else:  # TargetZone: the body the zone site is attached to (the table)
        sup_bodies = scene.subtree(site["body"]) | {site["body"]}
    sup = [c for c in o["env"] if _below_com(o, c)]
    sup_ok = [c for c in sup if int(scene.geom_body[c["other_g"]]) in sup_bodies]
    sup_other = [c for c in sup if int(scene.geom_body[c["other_g"]]) not in sup_bodies]
    com_m = hull_margin(hull, o["com"][:2])
    cfp = [hull_margin(hull, c["pos"][:2]) for c in sup_ok]
    support_top = max([c["pos"][2] for c in sup_ok], default=None)
    checks = {
        "not_held": not o["robot"],
        "at_rest": o["rest"],
        "supported_by_target": bool(sup_ok),
        "supported_only_by_target": bool(sup_ok) and not sup_other,
        "com_over_region": com_m is not None and com_m >= 0,
    }
    if pobj is None and support_top is not None:
        checks["resting_on_surface"] = o["low"] <= support_top + TOL["zone_lift_m"]
    _with_orientation(checks, o, "on_fixture_region" if pobj is not None else "on_table_zone")
    nums = _common_numbers(o)
    nums.update({"support_owner": parent or scene.owner(site["body"]), "supports_in_target": len(sup_ok),
                 "other_supports": sorted({f'{c["owner"]}:{c["other"]}' for c in sup_other}),
                 "com_margin_region_mm": r(com_m * 1000 if com_m is not None else None, 2),
                 "support_points_in_region": f"{sum(1 for m in cfp if m >= 0)}/{len(cfp)}",
                 "support_excluded_jointed_parts": sorted(scene.bname(b) for b in excluded),
                 "support_top_z": r(support_top)})
    return raw, lib, checks, nums


def explain_joint(scene, pred, name, window):
    core = scene.core
    raw = bool(core._eval_predicate([pred, name]))
    obj = core.get_object(name)
    props = obj.object_properties.get("articulation", {}) if hasattr(obj, "object_properties") else {}
    if name in core.object_sites_dict:
        props = core.get_object(core.object_sites_dict[name].parent_name).object_properties.get("articulation", {})
    info = [scene.joint_info(j) for j in joint_targets(scene, name)]
    part_bodies = set()
    for j in info:
        part_bodies |= scene.subtree(j["body"])
    robot = scene.robot_contacts(scene.geoms_of(part_bodies))
    rest = all(abs(j["qd"]) < TOL["rest_joint"] for j in info)
    if window is not None and info and all(len(window.jbuf.get(j["joint"], ())) >= 5 for j in info):
        rest = all((max(window.jbuf[j["joint"]]) - min(window.jbuf[j["joint"]])) < TOL["rest_joint_window"]
                   for j in info)
    for j in info:
        lo, hi = j["range"]
        cr = props.get("default_close_ranges") or props.get("default_turnoff_ranges")
        if cr:
            closed_end = lo if abs(np.mean(cr) - lo) < abs(np.mean(cr) - hi) else hi
            open_end = hi if closed_end == lo else lo
            j["fraction_from_closed"] = r((j["q"] - closed_end) / (open_end - closed_end), 4)
        j["q"], j["qd"] = r(j["q"], 5), r(j["qd"], 5)
    lib = {"tested": {"open": "ANY joint qpos past the open threshold",
                      "close": "ALL joints qpos past the close threshold",
                      "turnon": "ANY joint qpos >= min(turnon range)",
                      "turnoff": "ALL joints qpos < max(turnoff range)"}[pred],
           "articulation": jsonable(dict(props)), "joints": info}
    checks = {"libero_threshold": raw, "robot_not_touching_part": not robot, "joint_at_rest": rest}
    nums = {"robot_contacts": sorted({c["other"] for c in robot}),
            "part_bodies": sorted(scene.bname(b) for b in part_bodies)}
    return raw, lib, checks, nums


def explain(scene, atom, window=None):
    pred = atom[0].lower()
    core = scene.core
    if pred == "in":
        raw, lib, checks, nums = explain_in(scene, atom[1], atom[2], window)
        kind = "inside_container"
    elif pred == "on" and atom[2] in core.object_sites_dict:
        raw, lib, checks, nums = explain_on_site(scene, atom[1], atom[2], window)
        kind = "on_fixture_region" if core.object_sites_dict[atom[2]].parent_name else "on_table_zone"
    elif pred == "on":
        raw, lib, checks, nums = explain_on_object(scene, atom[1], atom[2], window)
        kind = "on_object"
    elif pred in ("open", "close", "turnon", "turnoff"):
        raw, lib, checks, nums = explain_joint(scene, pred, atom[1], window)
        kind = "articulation"
    else:
        return {"atom": list(atom), "kind": "unsupported", "raw": bool(core._eval_predicate(atom)), "strict": None,
                "why_not": ["unsupported_predicate"], "checks": {}, "libero": {}, "numbers": {}}
    if kind != "articulation":
        nums["orientation_required"] = kind in STRICT_ORIENTATION_KINDS
    return jsonable({"atom": list(atom), "kind": kind, "raw": raw, "strict": bool(all(checks.values())),
                     "why_not": [k for k, v in checks.items() if not v], "checks": {k: bool(v) for k, v in checks.items()},
                     "libero": lib, "numbers": nums})


def evaluate_goal(scene, window=None):
    atoms = [explain(scene, a, window) for a in scene.goal()]
    return {"truth_version": TRUTH_VERSION, "diagnosis_only": True, "affects_completion": False,
            "raw_conjunction": all(a["raw"] for a in atoms), "check_success": bool(scene.env.check_success()),
            "strict_conjunction": all(bool(a["strict"]) for a in atoms), "atoms": atoms,
            "window_steps": None if window is None else window.k}


# ---------------- counterfactual and verdict ----------------
def release_settle_actions(protocol=None):
    p = {**RELEASE_SETTLE, **(protocol or {})}
    opened = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    return ([list(opened)] * p["open_steps"]
            + [[0.0, 0.0, float(p["lift_action_z"]), 0.0, 0.0, 0.0, -1.0]] * p["lift_steps"]
            + [list(opened)] * p["settle_steps"])


def release_and_settle(env, monitors, *, scene=None, window=None, protocol=None, first_index=10_000_000):
    """Diagnosis-only counterfactual AFTER the episode outcome is fixed: open the gripper, lift the hand, let
    the scene settle. Never part of the scored episode, never traced as policy steps. Mutates the terminal state,
    so it must run after every terminal artifact is captured."""
    actions = release_settle_actions(protocol)
    raw_true = 0
    objs, joints = window_targets(scene) if scene is not None else ([], [])
    robot_touch = set()
    if window is not None:
        window.reset()
    for index, action in enumerate(actions):
        for monitor in monitors:
            monitor.begin_action(first_index + index)
        env.step(action)
        for position, monitor in enumerate(monitors):
            monitor.finish_action(**({"refresh": position == 0} if _accepts_refresh(monitor) else {}))
        raw_true += bool(env.check_success())
        if scene is not None and window is not None:
            window.update(scene, objs, joints)
            for o in objs:
                for c in scene.robot_contacts(scene.geoms_of(scene.object_bodies(o))):
                    robot_touch.add(f"{o}<->{c['other']}")
    return {"actions": len(actions), "protocol": {**RELEASE_SETTLE, **(protocol or {})},
            "libero_success_after_settle": bool(env.check_success()),
            "raw_true_fraction_during_settle": round(raw_true / max(1, len(actions)), 3),
            "robot_goal_object_contacts_during_settle": sorted(robot_touch)}


def _accepts_refresh(monitor):
    import inspect
    try:
        return "refresh" in inspect.signature(monitor.finish_action).parameters
    except (TypeError, ValueError):
        return False


def failing_checks(evaluation):
    """{atom json: [failing sub-checks]} for the atoms whose strict verdict is not True."""
    if not isinstance(evaluation, dict):
        return None
    out = {}
    for atom in evaluation.get("atoms") or []:
        if atom.get("strict") is not True:
            out[" ".join(map(str, atom.get("atom", [])))] = list(atom.get("why_not") or ["strict_unavailable"])
    return out


def held_only(failing):
    """True when EVERY failing atom fails only hold/rest consequences AND that atom itself has a hold check
    failing (per atom: an unheld moving object is never deferred because some other atom is held)."""
    if not failing:
        return False
    return all(set(checks) <= set(HELD_CHECKS) and bool(set(checks) & set(HOLD_CHECKS))
               for checks in failing.values())


def physical_summary(terminal, settled, *, raw_libero_success):
    """AC-2: summary.physical_truth. A v2 success needs raw LIBERO success AND strict physical truth at the
    terminal state; when the only failing terminal sub-checks come from the robot still holding the object,
    the release-and-settle counterfactual decides, and then LIBERO's raw success must also hold after the
    settle (strict is not a superset of raw: a release can move the origin out of LIBERO's box)."""
    at_terminal = bool(isinstance(terminal, dict) and terminal.get("strict_conjunction") is True)
    after_settle = bool(isinstance(settled, dict) and settled.get("strict_conjunction") is True)
    raw_after_settle = settled.get("check_success") if isinstance(settled, dict) else None
    failing = failing_checks(terminal)
    deferred = (not at_terminal) and held_only(failing)
    physical = at_terminal or (deferred and after_settle and raw_after_settle is True)
    raw = raw_libero_success is True
    if terminal is None:
        verdict = "truth_unavailable"
    elif raw and physical:
        verdict = "physical_success"
    elif raw:
        verdict = "lenient_success"
    elif physical:
        verdict = "raw_failure_strict_holds"
    else:
        verdict = "raw_failure"
    return {"at_terminal": terminal is not None and at_terminal,
            "after_release_settle": settled is not None and after_settle,
            "physical_success_at_terminal": at_terminal,
            "physical_success_after_settle": after_settle,
            "terminal_failing_checks": failing, "terminal_failure_is_hold_only": deferred,
            "raw_libero_success_after_settle": raw_after_settle,
            "physical_success": physical, "raw_libero_success": raw_libero_success,
            "v2_success": raw and physical, "verdict": verdict,
            "diagnosis_only": True, "affects_completion": False, "truth_version": TRUTH_VERSION}


def answer_key(scene):
    """Evaluator-only answer key: BDDL goal atoms, objects, fixtures and regions (never shown to the policy)."""
    core = scene.core
    regions = {}
    for name, site in core.object_sites_dict.items():
        regions[name] = {"parent": getattr(site, "parent_name", None),
                         "joints": list(getattr(site, "joints", None) or [])}
    objects = {name: getattr(obj, "category", None) or type(obj).__name__ for name, obj in core.objects_dict.items()}
    fixtures = {name: getattr(obj, "category", None) or type(obj).__name__ for name, obj in core.fixtures_dict.items()}
    try:
        language = core.parsed_problem.get("language_instruction")
    except Exception:  # pragma: no cover - diagnostic only
        language = None
    if isinstance(language, (list, tuple)):
        language = " ".join(map(str, language))
    return jsonable({"diagnosis_only": True, "goal_atoms": scene.goal(), "objects": objects, "fixtures": fixtures,
                     "regions": regions, "language": language})


PLAN_RELATION_FOR_PREDICATE = {"turnon": "turn", "turnoff": "turn", "open": "slide_or_turn_open",
                               "close": "slide_or_turn_close", "on": "place_on_or_push_to", "in": "place_in"}


def parse_check(answer, plan=None):
    """Plan relations against the BDDL atoms (§6.5). M0 has no v2 plan: every atom is reported as unplanned."""
    atoms = (answer or {}).get("goal_atoms") or []
    return {"diagnosis_only": True, "plan_available": plan is not None,
            "atoms": [{"atom": atom, "expected_relation": PLAN_RELATION_FOR_PREDICATE.get(str(atom[0]).lower()),
                       "planned": None} for atom in atoms],
            "note": None if plan is not None else "no v2 relation plan exists before M5/M6"}
