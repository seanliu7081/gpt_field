"""Create frozen scene state, previews and evaluator-only validation; no policy."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from load_task import HERE, HOLD, LIBERO_ROOT, PROMPT, SEED, SETTLE_STEPS, TASK_KEY, EXTERNAL_FOVY, load_task

ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
from gpt_field.observation import PolicyObservation

BOWL = "akita_black_bowl_1"
POT = "moka_pot_1"
CABINET = "wooden_cabinet_1"
STOVE = "flat_stove_1"


def write(name, value):
    def native(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=native,
                                      allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helpers():
    path = HERE.parent / "libero_10_t8_bottle_obstacle_20261103_001/prepare_layout.py"
    spec = importlib.util.spec_from_file_location("scene_render_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module, path


def place_upright(env, helper, name, xy, surface_z):
    joint = env.env.objects_dict[name].joints[0]
    env.sim.data.set_joint_qpos(joint, [*xy, 1., 1., 0., 0., 0.])
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()
    relative_bottom = helper.geom_bounds(env, name)["lower_xyz_m"][2] - 1.
    env.sim.data.set_joint_qpos(joint, [*xy, surface_z-relative_bottom+.0002, 1., 0., 0., 0.])
    env.sim.forward()


def build(helper):
    with load_task(bootstrap=True, ordered=False) as (env, raw, prompt):
        assert prompt == PROMPT
        # Reuse the previous t08 robot reset by joint name, while all scene
        # objects and fixtures belong to this new task.
        source_dir = HERE.parent / "libero_10_t8_bottle_obstacle_20261103_001"
        source_state = np.load(source_dir / "original_initial_state.npy", allow_pickle=False)
        mappings = json.loads((source_dir / "layout_manifest.json").read_text())["mapped_joints"]
        robot_mapping = []
        for item in mappings:
            name = item["joint"]
            if not name.startswith(("robot0_", "gripper0_")):
                continue
            qa, qb = item["source_qpos_address"]
            va, vb = item["source_qvel_address"]
            assert qb-qa == vb-va == 1
            env.sim.data.set_joint_qpos(name, source_state[1+qa])
            env.sim.data.set_joint_qvel(name, source_state[1+24+va])
            robot_mapping.append(name)
        for level in ("top", "middle", "bottom"):
            env.sim.data.set_joint_qpos(f"{CABINET}_{level}_level", .001)
            env.sim.data.set_joint_qvel(f"{CABINET}_{level}_level", 0.)
        env.sim.data.set_joint_qpos(f"{STOVE}_button", -.0025)
        env.sim.data.set_joint_qvel(f"{STOVE}_button", 0.)
        env.sim.forward()
        m, d = env.sim.model, env.sim.data
        # Find the actual drawer floor rather than relying on inaccurate asset
        # bottom_site offsets (the scanned bowl's offset is not its physical base).
        bid = m.body_name2id(f"{CABINET}_cabinet_top")
        floors = []
        for i in range(m.ngeom):
            if m.geom_bodyid[i] != bid or m.geom_type[i] != 6 or not (m.geom_contype[i] or m.geom_conaffinity[i]):
                continue
            ext = np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i]
            if ext[2] < .01 and min(ext[:2]) > .05:
                floors.append((float(d.geom_xpos[i, 2]+ext[2]), i))
        assert len(floors) == 1, floors
        floor_z, floor_id = floors[0]
        sid = m.site_name2id(f"{CABINET}_top_region")
        bowl_xy = d.site_xpos[sid, :2].copy()
        place_upright(env, helper, BOWL, bowl_xy, floor_z)
        table = helper.table_surface(env, [.02, .12])
        place_upright(env, helper, POT, [.02, .12], table["upper_z_m"])
        state = np.asarray(env.sim.get_state().flatten()).copy()
        np.save(HERE / "initial_state.npy", state, allow_pickle=False)
        write("construction.json", {
            "seed": SEED, "right_definition": "positive world Y, right in upright agentview preview",
            "cabinet_origin_xy_m": [-.08, .38], "cabinet_yaw_rad": -np.pi/2,
            "drawer_open_direction_world": [-1., 0., 0.],
            "stove_origin_xy_m": [-.34, .12], "moka_center_xy_m": [.02, .12],
            "drawer_floor_z_m": floor_z, "drawer_floor_geom": m.geom_id2name(floor_id),
            "table_support": table, "initial_object_poses": helper.object_poses(env),
            "nq": m.nq, "nv": m.nv, "na": m.na,
            "robot_reset_source": str(source_dir / "original_initial_state.npy"),
            "robot_reset_source_sha256": sha(source_dir / "original_initial_state.npy"),
            "mapped_robot_joints": robot_mapping,
        })


def top_down(env, prefix="custom_"):
    """Evaluator-only top view: world +Y stays image-right, robot at the top."""
    from PIL import Image
    m = env.sim.model
    cid = m.camera_name2id("agentview")
    old = m.cam_pos[cid].copy(), m.cam_quat[cid].copy(), float(m.cam_fovy[cid])
    try:
        m.cam_pos[cid] = [-.06, .10, 2.35]
        m.cam_quat[cid] = [2**-.5, 0., 0., 2**-.5]
        m.cam_fovy[cid] = 52.
        env.sim.forward()
        frame = env.sim.render(camera_name="agentview", width=1024, height=1024)
        Image.fromarray(np.ascontiguousarray(frame[::-1])).save(HERE / f"{prefix}top_down.png")
    finally:
        m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = old
        env.sim.forward()


def main():
    helper, helper_path = helpers()
    build(helper)
    with load_task(ordered=False) as (env, raw, prompt):
        obs = PolicyObservation.from_libero(raw, prompt, image_size=512).to_dict()
        digest = initial_observation_sha256(obs)
        write("initial_observation.json", obs)
        (HERE / "initial_observation.sha256").write_text(digest+"\n")
        helper.images(obs, "custom_")
        helper.external_highres(env, "custom_")
        top_down(env)
        initial_state = np.asarray(env.sim.get_state().flatten()).copy()
        np.save(HERE / "settled_state.npy", initial_state, allow_pickle=False)
        from validate_scene import validate_scene
        validation = validate_scene(env)
        write("scene_validation.json", validation)
        assert np.array_equal(initial_state, np.asarray(env.sim.get_state().flatten()))
        # Evaluator-only reveal: opened drawer + translated contained bowl. This
        # is a layout preview, never the task reset or an agent action trace.
        j = f"{CABINET}_top_level"
        before = float(env.sim.data.get_joint_qpos(j))
        bowl_joint = env.env.objects_dict[BOWL].joints[0]
        bowl_pose = np.asarray(env.sim.data.get_joint_qpos(bowl_joint)).copy()
        env.sim.data.set_joint_qpos(j, -.16)
        env.sim.forward()
        axis = env.sim.data.xaxis[env.sim.model.joint_name2id(j)].copy()
        bowl_pose[:3] += axis*(-.16-before)
        env.sim.data.set_joint_qpos(bowl_joint, bowl_pose)
        env.sim.forward()
        helper.external_highres(env, "drawer_open_diagnostic_")
        top_down(env, "drawer_open_diagnostic_")
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
    with load_task(ordered=False) as (env, raw, prompt):
        repeat = initial_observation_sha256(PolicyObservation.from_libero(raw, prompt, image_size=512).to_dict())
    checks = {
        "scene_validation_passed": validation["all_passed"],
        "repeat_reset_observation_identical": repeat == digest,
    }
    manifest = {
        "kind": "custom_composite_task", "task_key": TASK_KEY, "prompt": PROMPT,
        "seed": SEED, "episode_index": 0, "settle_steps": SETTLE_STEPS,
        "right_definition": "right in upright external agentview preview (world +Y)",
        "initial_state": {"drawer": "closed", "bowl": "inside top drawer", "stove": "off", "moka_pot": "on table"},
        "ordered_steps": ["open top drawer", "place bowl on cabinet top", "close top drawer", "turn stove on", "place moka pot on stove"],
        "added_obstacles": [], "movable_objects": [BOWL, POT], "fixtures": [CABINET, STOVE],
        "model_calls": 0, "policy_actions": 0,
        "validation_scope": "Passive simulator checks and evaluator-only counterfactuals; not an agent completion test.",
        "simulator_configuration": {"libero_root": str(LIBERO_ROOT), "python_executable": sys.executable,
                                    "controller": "OSC_POSE", "control_freq_hz": 20, "render_size": 512,
                                    "external_camera_fovy_deg": EXTERNAL_FOVY},
        "initial_observation_sha256": digest, "repeat_initial_observation_sha256": repeat,
        "file_sha256": {name: sha(HERE/name) for name in ("custom_task.bddl", "initial_state.npy", "settled_state.npy", "load_task.py", "prepare_layout.py", "validate_scene.py")},
        "reused_render_helper": {"path": str(helper_path), "sha256": sha(helper_path)},
        "checks": checks, "checks_passed": all(checks.values()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write("layout_manifest.json", manifest)
    assert all(checks.values()), checks
    print(json.dumps({"task_key": TASK_KEY, "checks": checks, "observation_sha256": digest}))


if __name__ == "__main__":
    main()
