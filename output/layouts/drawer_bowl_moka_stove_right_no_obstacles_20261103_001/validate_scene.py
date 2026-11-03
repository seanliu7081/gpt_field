"""Independent evaluator checks for the drawer/bowl and stove/moka scene.

This module never calls a policy. Counterfactual checks change disposable
simulator states, use the real LIBERO predicates, and restore the input state.
The passive stability check uses ten zero-motion, open-gripper settle actions.
"""
from __future__ import annotations

from itertools import product

import numpy as np

BOWL = "akita_black_bowl_1"
POT = "moka_pot_1"
CABINET = "wooden_cabinet_1"
STOVE = "flat_stove_1"
DRAWER = "wooden_cabinet_1_top_level"
BUTTON = "flat_stove_1_button"
INSIDE = "wooden_cabinet_1_top_region"
TOP = "wooden_cabinet_1_top_side"
COOK = "flat_stove_1_cook_region"
HOLD = [0., 0., 0., 0., 0., 0., -1.]
CLOSED_QPOS = .001  # Positive stock Close predicate, without shelf self-penetration.


def _raw(env):
    return env.env


def _pred(env, *args):
    return bool(_raw(env)._eval_predicate(list(args)))


def _state(env):
    return np.asarray(env.sim.get_state().flatten()).copy()


def _restore(env, state):
    env.set_init_state(state)


def _object(env, name):
    raw = _raw(env)
    return raw.objects_dict.get(name) or raw.fixtures_dict[name]


def _pose(env, name):
    return np.asarray(env.sim.data.get_joint_qpos(_object(env, name).joints[0])).copy()


def _set_pose(env, name, pose):
    joint = _object(env, name).joints[0]
    env.sim.data.set_joint_qpos(joint, pose)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))


def _bounds(env, gid):
    model, data = env.sim.model, env.sim.data
    if int(model.geom_type[gid]) != 6:
        raise ValueError("Validation expects the unchanged stock box collision geometry")
    ext = np.abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
    return data.geom_xpos[gid] - ext, data.geom_xpos[gid] + ext


def _ids(env, name):
    return [env.sim.model.geom_name2id(g) for g in _object(env, name).contact_geoms]


def _object_bounds(env, name):
    values = [_bounds(env, gid) for gid in _ids(env, name)]
    return np.min([v[0] for v in values], axis=0), np.max([v[1] for v in values], axis=0)


def _owner(env, gid):
    name = env.sim.model.geom_id2name(gid) or ""
    for obj in (BOWL, POT, CABINET, STOVE):
        if name.startswith(obj + "_"):
            return obj
    if name.startswith(("robot", "gripper")):
        return "robot"
    return "table_or_arena"


def _contacts(env):
    result = []
    for c in env.sim.data.contact[:env.sim.data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        result.append({"a": _owner(env, a), "b": _owner(env, b),
                       "geom_a": env.sim.model.geom_id2name(a),
                       "geom_b": env.sim.model.geom_id2name(b),
                       "distance_m": float(c.dist),
                       "normal": np.asarray(c.frame[:3]).tolist()})
    return result


def _support_contacts(env, name, support):
    matches = []
    for c in _contacts(env):
        if name not in (c["a"], c["b"]):
            continue
        other = c["b"] if c["a"] == name else c["a"]
        upward = -c["normal"][2] if c["a"] == name else c["normal"][2]
        if other == support and upward > .95:
            matches.append(c)
    return matches


def _support_pose(env, name, support_gid, site):
    pose = _pose(env, name)
    lower, _ = _object_bounds(env, name)
    _, high = _bounds(env, support_gid)
    base_offset = pose[2] - lower[2]
    pose[:2] = env.sim.data.get_site_xpos(site)[:2]
    pose[2] = high[2] + base_offset - 1e-6
    return pose


def _roof_geometry(env):
    """Select the actual thin horizontal cabinet roof under top_side."""
    z = float(env.sim.data.get_site_xpos(TOP)[2])
    candidates = []
    for gid in _ids(env, CABINET):
        low, high = _bounds(env, gid)
        span = high - low
        if span[2] < .012 and min(span[:2]) > .10 and abs((low[2] + high[2]) / 2 - z) < .01:
            candidates.append(gid)
    if len(candidates) != 1:
        raise ValueError(f"Expected one cabinet roof collision box, found {candidates}")
    return candidates[0]


def _drawer_axis(env):
    model, data = env.sim.model, env.sim.data
    jid = model.joint_name2id(DRAWER)
    bid = int(model.jnt_bodyid[jid])
    return data.body_xmat[bid].reshape(3, 3) @ model.jnt_axis[jid]


def _goals(env):
    return {"bowl_on_cabinet_top": _pred(env, "on", BOWL, TOP),
            "drawer_closed": _pred(env, "close", INSIDE),
            "stove_on": _pred(env, "turnon", STOVE),
            "moka_pot_on_stove": _pred(env, "on", POT, COOK)}


def validate_goals(env):
    """All sixteen conjunction cases against stock LIBERO, without integration."""
    original = _state(env)
    original_bowl = _pose(env, BOWL)
    original_drawer = float(env.sim.data.get_joint_qpos(DRAWER))
    axis = _drawer_axis(env)
    roof = _roof_geometry(env)
    burner = [g for g in _ids(env, STOVE)
              if "collision_burner" in env.sim.model.geom_id2name(g)]
    if len(burner) != 1:
        raise ValueError("Expected the stock stove burner collision box")
    bowl_on = _support_pose(env, BOWL, roof, TOP)
    pot_on = _support_pose(env, POT, burner[0], COOK)
    rows = []
    try:
        for bowl_top, closed, on, pot_stove in product((False, True), repeat=4):
            _restore(env, original)
            drawer_q = CLOSED_QPOS if closed else -.15
            env.sim.data.set_joint_qpos(DRAWER, drawer_q)
            env.sim.data.set_joint_qvel(DRAWER, 0.)
            env.sim.data.set_joint_qpos(BUTTON, 1.0 if on else -.0025)
            env.sim.data.set_joint_qvel(BUTTON, 0.)
            bowl = bowl_on.copy() if bowl_top else original_bowl.copy()
            if not bowl_top:
                bowl[:3] += axis * (drawer_q - original_drawer)
            _set_pose(env, BOWL, bowl)
            if pot_stove:
                _set_pose(env, POT, pot_on)
            _restore(env, _state(env))
            expected = dict(zip(_goals(env), (bowl_top, closed, on, pot_stove)))
            actual = _goals(env)
            success = bool(env.check_success())
            rows.append({"expected_predicates": expected, "actual_predicates": actual,
                         "expected_success": all(expected.values()), "actual_success": success,
                         "passed": actual == expected and success == all(expected.values())})
    finally:
        _restore(env, original)
    restored = bool(np.array_equal(original, _state(env)))
    return {"kind": "evaluator_only_counterfactual_goal_truth_table", "rows": rows,
            "model_calls": 0, "policy_actions": 0, "physics_integration_steps": 0,
            "counterfactual_on_poses": {BOWL: bowl_on.tolist(), POT: pot_on.tolist()},
            "support_geometries": {BOWL: env.sim.model.geom_id2name(roof),
                                   POT: env.sim.model.geom_id2name(burner[0])},
            "state_restored_bitwise": restored,
            "all_passed": len(rows) == 16 and all(r["passed"] for r in rows) and restored,
            "scope": "Final conjunction only; ordered stage completion is validated separately."}


def validate_drawer_sweep(env):
    """Check sampled kinematic drawer travel with its contained bowl translated."""
    original = _state(env)
    bowl_original = _pose(env, BOWL)
    drawer_original = float(env.sim.data.get_joint_qpos(DRAWER))
    axis = _drawer_axis(env)
    rows = []
    try:
        for q in np.linspace(CLOSED_QPOS, -.16, 14):
            _restore(env, original)
            env.sim.data.set_joint_qpos(DRAWER, float(q))
            env.sim.data.set_joint_qvel(DRAWER, 0.)
            bowl = bowl_original.copy()
            bowl[:3] += axis * (q - drawer_original)
            _set_pose(env, BOWL, bowl)
            _restore(env, _state(env))
            forbidden = [c for c in _contacts(env)
                         if ({c["a"], c["b"]} & {CABINET, BOWL})
                         and ({c["a"], c["b"]} & {STOVE, POT, "robot"})]
            rows.append({"drawer_qpos_m": float(q), "forbidden_contacts": forbidden,
                         "bowl_inside_drawer": _pred(env, "in", BOWL, INSIDE),
                         "passed": not forbidden and _pred(env, "in", BOWL, INSIDE)})
    finally:
        _restore(env, original)
    restored = bool(np.array_equal(original, _state(env)))
    return {"kind": "sampled_evaluator_kinematic_clearance_check", "rows": rows,
            "drawer_world_positive_axis": axis.tolist(), "state_restored_bitwise": restored,
            "physics_integration_steps": 0, "policy_actions": 0, "model_calls": 0,
            "all_passed": all(r["passed"] for r in rows) and restored,
            "scope": "Fourteen sampled positions; does not prove a robot grasp/pull trajectory."}


def validate_scene(env, passive_steps=10):
    """Validate a settled raw OffScreenRenderEnv; restore its input state."""
    original = _state(env)
    raw = _raw(env)
    initial_bowl, initial_pot = _pose(env, BOWL), _pose(env, POT)
    before = {BOWL: initial_bowl, POT: initial_pot}
    predicates = {"bowl_inside_drawer": _pred(env, "in", BOWL, INSIDE),
                  "drawer_closed": _pred(env, "close", INSIDE),
                  "stove_off": _pred(env, "turnoff", STOVE),
                  **_goals(env), "initial_success": bool(env.check_success())}
    bowl_bounds = _object_bounds(env, BOWL)
    roof_low, _ = _bounds(env, _roof_geometry(env))
    roof_clearance = float(roof_low[2] - bowl_bounds[1][2])
    sid = env.sim.model.site_name2id(INSIDE)
    site_pos = env.sim.data.site_xpos[sid]
    extent = np.abs(env.sim.data.site_xmat[sid].reshape(3, 3)) @ env.sim.model.site_size[sid]
    cavity_low, cavity_high = site_pos - extent, site_pos + extent
    lateral_inside = bool(np.all(bowl_bounds[0][:2] > cavity_low[:2])
                          and np.all(bowl_bounds[1][:2] < cavity_high[:2]))
    supports = {BOWL: _support_contacts(env, BOWL, CABINET),
                POT: _support_contacts(env, POT, "table_or_arena")}
    initial_contacts = _contacts(env)
    cabinet_penetrations = [c for c in initial_contacts
                           if c["a"] == c["b"] == CABINET and c["distance_m"] < -1e-5]
    forbidden_pairs = ({BOWL, POT}, {BOWL, STOVE}, {POT, STOVE}, {POT, CABINET},
                       {CABINET, STOVE}, *({"robot", name} for name in (BOWL, POT, CABINET, STOVE)))
    forbidden_contacts = [c for c in initial_contacts if {c["a"], c["b"]} in forbidden_pairs]
    objects = sorted(raw.objects_dict)
    fixtures = sorted(set(raw.fixtures_dict) - {"kitchen_table"})
    centers = {name: env.sim.data.body_xpos[env.sim.model.body_name2id(_object(env, name).root_body)].tolist()
               for name in (CABINET, STOVE, BOWL, POT)}
    checks = {
        "exactly_two_task_objects_no_distractors": objects == sorted([BOWL, POT]),
        "exactly_cabinet_and_stove_fixtures": fixtures == sorted([CABINET, STOVE]),
        "all_task_centers_image_right_positive_y": all(p[1] > 0 for p in centers.values()),
        "initial_bowl_inside_closed_drawer": predicates["bowl_inside_drawer"] and predicates["drawer_closed"],
        "initial_stove_off": predicates["stove_off"] and not predicates["stove_on"],
        "initial_objects_not_already_on_targets": not predicates["bowl_on_cabinet_top"] and not predicates["moka_pot_on_stove"],
        "initial_goal_false": not predicates["initial_success"],
        "bowl_below_cabinet_roof_without_intersection": roof_clearance > .001,
        "whole_bowl_fits_drawer_laterally": lateral_inside,
        "bowl_supported_by_cabinet": bool(supports[BOWL]),
        "moka_supported_by_table": bool(supports[POT]),
        "no_initial_cabinet_self_penetration": not cabinet_penetrations,
        "no_initial_unwanted_task_or_robot_contacts": not forbidden_contacts,
    }
    passive = {}
    try:
        for _ in range(passive_steps):
            env.step(HOLD)
        after = {name: _pose(env, name) for name in before}
        for name in before:
            angle = float(2 * np.arccos(np.clip(abs(np.dot(before[name][3:], after[name][3:])), 0, 1)))
            distance = float(np.linalg.norm(after[name][:3] - before[name][:3]))
            passive[name] = {"translation_drift_m": distance, "rotation_drift_rad": angle,
                             "before_qpos": before[name].tolist(), "after_qpos": after[name].tolist()}
            checks[name + "_passive_stable_under_2mm_and_1deg"] = bool(distance < .002 and angle < np.deg2rad(1.))
        checks["passive_hold_preserves_required_init_predicates"] = (
            _pred(env, "in", BOWL, INSIDE) and _pred(env, "close", INSIDE)
            and _pred(env, "turnoff", STOVE) and not env.check_success())
    finally:
        _restore(env, original)
    goals = validate_goals(env)
    sweep = validate_drawer_sweep(env)
    checks["sixteen_goal_truth_table_cases_passed"] = goals["all_passed"]
    checks["sampled_drawer_sweep_clear"] = sweep["all_passed"]
    checks["state_restored_bitwise"] = bool(np.array_equal(original, _state(env)))
    return {"kind": "independent_custom_scene_validation", "checks": checks,
            "all_passed": all(checks.values()), "model_calls": 0, "policy_actions": 0,
            "passive_hold_actions": passive_steps, "objects": objects, "fixtures": fixtures,
            "initial_predicates": predicates, "task_body_centers_m": centers,
            "bowl_roof_clearance_m": roof_clearance,
            "initial_support_contacts": supports, "passive_stability": passive,
            "initial_cabinet_self_penetrations": cabinet_penetrations,
            "initial_unwanted_contacts": forbidden_contacts,
            "goal_truth_table": goals, "drawer_sweep": sweep,
            "scope": "Scene geometry, reset stability, physical predicates, and sampled drawer travel; no policy completion test."}


if __name__ == "__main__":
    import json
    from pathlib import Path
    from load_task import load_task

    with load_task(ordered=False) as (env, _, _):
        result = validate_scene(env)
    path = Path(__file__).with_name("independent_scene_validation.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": result["all_passed"], "checks": result["checks"]}))
    if not result["all_passed"]:
        raise SystemExit(1)
