"""Check the task's actual LIBERO predicates using disposable simulator states.

These are evaluator-only logical counterfactuals, not robot demonstrations or
policy rollouts. The incoming settled simulator state is restored before return.
"""

from itertools import product


POTS = ("moka_pot_1", "moka_pot_2")
STOVE = "flat_stove_1"
COOK_REGION = "flat_stove_1_cook_region"
BUTTON = "flat_stove_1_button"


def _box_bounds(env, geom_id):
    import numpy as np

    model, data = env.sim.model, env.sim.data
    if int(model.geom_type[geom_id]) != 6:
        raise ValueError("Goal validation expects the stock box collision geometry")
    extent = np.abs(data.geom_xmat[geom_id].reshape(3, 3)) @ model.geom_size[geom_id]
    return data.geom_xpos[geom_id] - extent, data.geom_xpos[geom_id] + extent


def _support_poses(env):
    """Place the upright pots side by side on the actual burner collision box."""
    import numpy as np

    model, data = env.sim.model, env.sim.data
    stove = env.env.fixtures_dict[STOVE]
    support_ids = [model.geom_name2id(name) for name in stove.contact_geoms
                   if "collision_burner" in name]
    if len(support_ids) != 1:
        raise ValueError("Expected one stock stove burner support collision box")
    support_low, support_high = _box_bounds(env, support_ids[0])
    center = np.asarray(data.get_site_xpos(COOK_REGION)).copy()
    halfsize = np.asarray(env.env.object_sites_dict[COOK_REGION].size)
    poses = {}
    for name, sign in zip(POTS, (-1, 1)):
        obj = env.env.objects_dict[name]
        qpos = np.asarray(data.get_joint_qpos(obj.joints[0])).copy()
        lower_z = min(_box_bounds(env, model.geom_name2id(geom))[0][2]
                      for geom in obj.contact_geoms)
        # Keep the incoming upright quaternion. The collision geometry supplies
        # the body-to-base offset; neither a visual mesh nor bottom_site does.
        body_to_base = qpos[2] - lower_z
        qpos[:2] = center[:2] + np.array([sign * halfsize[0] * 0.6, 0.0])
        qpos[2] = support_high[2] + body_to_base - 1e-6
        if not np.all((qpos[:2] > support_low[:2]) & (qpos[:2] < support_high[:2])):
            raise ValueError("Counterfactual pot center lies outside burner support")
        poses[name] = qpos
    return poses, {"geom": model.geom_id2name(support_ids[0]),
                   "top_z_m": float(support_high[2]), "contact_overlap_m": 1e-6}


def validate_goals(env):
    """Return all eight predicate cases, preserving the incoming simulator state.

    ``env`` must be the new task's settled raw ``OffScreenRenderEnv``. No action,
    model call, controller change, or physics integration is performed here.
    """
    import numpy as np

    original = np.asarray(env.sim.get_state().flatten()).copy()
    rows = []
    support = None
    support_poses = {}
    try:
        support_poses, support = _support_poses(env)
        for stove_on, pot1_on, pot2_on in product((False, True), repeat=3):
            env.set_init_state(original)
            data = env.sim.data
            data.set_joint_qpos(BUTTON, 1.0 if stove_on else -0.0025)
            data.set_joint_qvel(BUTTON, 0.0)
            expected = {"stove_on": stove_on, "moka_pot_1_on_stove": pot1_on,
                        "moka_pot_2_on_stove": pot2_on}
            for name, on_stove in zip(POTS, (pot1_on, pot2_on)):
                if on_stove:
                    joint = env.env.objects_dict[name].joints[0]
                    data.set_joint_qpos(joint, support_poses[name])
                    data.set_joint_qvel(joint, np.zeros(6))
            env.set_init_state(np.asarray(env.sim.get_state().flatten()).copy())
            predicates = {
                "stove_on": bool(env.env._eval_predicate(["turnon", STOVE])),
                **{name + "_on_stove": bool(env.env._eval_predicate(
                    ["on", name, COOK_REGION])) for name in POTS},
            }
            success = bool(env.check_success())
            expected_success = all(expected.values())
            rows.append({"expected_predicates": expected,
                         "actual_predicates": predicates,
                         "expected_success": expected_success,
                         "actual_success": success,
                         "button_qpos": float(data.get_joint_qpos(BUTTON)),
                         "passed": predicates == expected and success == expected_success})
    finally:
        env.set_init_state(original)
    restored = np.array_equal(original, np.asarray(env.sim.get_state().flatten()))
    return {"kind": "evaluator_only_counterfactual_goal_truth_table",
            "policy_actions": 0, "model_calls": 0,
            "physics_integration_steps": 0,
            "rows": rows, "state_restored_bitwise": bool(restored),
            "support_geometry": support,
            "counterfactual_on_poses": {name: pose.tolist()
                                       for name, pose in support_poses.items()},
            "all_passed": len(rows) == 8 and all(row["passed"] for row in rows) and bool(restored)}
