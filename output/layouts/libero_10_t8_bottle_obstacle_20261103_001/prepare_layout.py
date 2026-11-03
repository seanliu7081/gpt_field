"""Evaluator-only t08 scene authoring and previews; no model or policy rollout.

Run with any Python: this script starts separate native simulator subprocesses
on EGL device 3. The custom BDDL's bootstrap region is used only while building
the model; initial_state.npy is the authoritative named-joint mapped layout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM_PYTHON = "/workspace/venvs/libero-plus/bin/python"
CAN = "ketchup_obstacle_1"
sys.path.insert(0, str(ROOT / "src"))


def write(name, value):
    import numpy as np
    def native(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False, default=native) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(custom=False):
    from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config
    cfg, monitor = build_v2_sim_config("libero_10_t8", sim_python=SIM_PYTHON)
    if custom:
        cfg = replace(cfg, task_bddl_override=str(HERE / "custom_task.bddl"),
                      initial_state_override=str(HERE / "initial_state.npy"))
    return cfg, monitor


def images(value, prefix):
    from PIL import Image
    from gpt_field.contact_skills import StepView
    from gpt_field.observation import PolicyObservation
    view = StepView(PolicyObservation.from_dict(value))
    for name, pixels in (("external", view.external_rgb_upright), ("wrist", view.wrist_rgb_upright)):
        Image.fromarray(pixels).save(HERE / f"{prefix}{name}.png")


def address(model, name, velocity=False):
    raw = (model.get_joint_qvel_addr if velocity else model.get_joint_qpos_addr)(name)
    return tuple(int(v) for v in raw) if isinstance(raw, tuple) else (int(raw), int(raw) + 1)


def owners(env):
    model = env.sim.model
    roots = {model.body_name2id(obj.root_body): name for name, obj in
             {**env.env.objects_dict, **env.env.fixtures_dict}.items()}
    def owner(geom):
        body = int(model.geom_bodyid[geom])
        while body:
            if body in roots:
                return roots[body]
            name = model.body_id2name(body) or ""
            if name.startswith(("robot", "gripper")):
                return "robot"
            body = int(model.body_parentid[body])
        return "world"
    return owner


def contacts(env):
    model, data = env.sim.model, env.sim.data
    owner = owners(env)
    result = []
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = owner(int(c.geom1)), owner(int(c.geom2))
        if a == b or c.dist > 1e-6:
            continue
        result.append({"a": a, "b": b, "distance_m": float(c.dist),
                       "geom1": model.geom_id2name(int(c.geom1)),
                       "geom2": model.geom_id2name(int(c.geom2)),
                       "position_xyz_m": c.pos.tolist(), "normal": c.frame[:3].tolist()})
    return result


def object_poses(env):
    import numpy as np
    result = {}
    for name, obj in env.env.objects_dict.items():
        bid = env.sim.model.body_name2id(obj.root_body)
        result[name] = {"joint": obj.joints[0],
            "qpos": np.asarray(env.sim.data.get_joint_qpos(obj.joints[0])).tolist(),
            "qvel": np.asarray(env.sim.data.get_joint_qvel(obj.joints[0])).tolist(),
            "body_xyz_m": env.sim.data.body_xpos[bid].tolist()}
    return result


def geom_bounds(env, name):
    import numpy as np
    m, d = env.sim.model, env.sim.data
    owner = owners(env)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    ids = []
    for i in range(m.ngeom):
        if owner(i) != name or not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        if int(m.geom_type[i]) != 6:
            raise ValueError("Can collision bounds expect the stock box decomposition")
        extent = np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i]
        low = np.minimum(low, d.geom_xpos[i] - extent)
        high = np.maximum(high, d.geom_xpos[i] + extent)
        ids.append(i)
    return {"lower_xyz_m": low.tolist(), "upper_xyz_m": high.tolist(),
            "size_xyz_m": (high - low).tolist(), "collision_box_count": len(ids)}


def table_surface(env, xy):
    """Use compiled support geometry; workspace_offset is not the tabletop."""
    import numpy as np
    m, d = env.sim.model, env.sim.data
    surfaces = []
    for i in range(m.ngeom):
        body = int(m.geom_bodyid[i])
        if ((m.geom_id2name(i) or "") != "table_collision"
                or int(m.geom_type[i]) != 6 or not (m.geom_contype[i] or m.geom_conaffinity[i])):
            continue
        extent = np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i]
        low, high = d.geom_xpos[i] - extent, d.geom_xpos[i] + extent
        if np.all(np.asarray(xy) >= low[:2]) and np.all(np.asarray(xy) <= high[:2]):
            surfaces.append({"geom_id": int(i), "body": m.body_id2name(body),
                             "upper_z_m": float(high[2]), "lower_xyz_m": low.tolist(),
                             "upper_xyz_m": high.tolist()})
    return max(surfaces, key=lambda item: item["upper_z_m"])


def build():
    import numpy as np
    import torch
    from gpt_field.agent_speedup_v2_user_check_sim import UserCheckLiberoWorker
    from gpt_field.libero import validate_task_bddl_contract
    cfg, monitor = config()
    worker = UserCheckLiberoWorker(cfg, monitor)
    custom_env = None
    try:
        original_response = worker.reset(8, 0)
        images(original_response["observation"], "original_")
        write("original_observation.json", original_response["observation"])
        original_settled = object_poses(worker.env)
        original_fixture = fixture_snapshot(worker.env)
        external_highres(worker.env, "original_")
        top_down(worker.env, "original_")
        settled_original_state = np.asarray(worker.env.sim.get_state().flatten()).copy()
        task = worker.benchmark.get_task(8)
        source = worker.benchmark_root / "init_files" / task.problem_folder / task.init_states_file
        original = np.asarray(torch.load(source, map_location="cpu", weights_only=False)[0]).copy()
        np.save(HERE / "original_initial_state.npy", original, allow_pickle=False)
        original_bddl = Path(worker.benchmark.get_task_bddl_file_path(8))
        (HERE / "original_task.bddl").write_bytes(original_bddl.read_bytes())
        text = original_bddl.read_text()
        text = text.replace("(:regions", """(:regions
      (bottle_bootstrap_region
        (:target kitchen_table)
        (:ranges ((0.20 -0.30 0.30 -0.18)))
        (:yaw_rotation ((0.0 0.0))))""", 1)
        text = text.replace("(:objects", f"(:objects\n    {CAN} - ketchup", 1)
        text = text.replace("    (Turnon flat_stove_1)", f"    (Turnon flat_stove_1)\n    (On {CAN} kitchen_table_bottle_bootstrap_region)", 1)
        (HERE / "custom_task.bddl").write_text(text)
        contract = validate_task_bddl_contract(original_bddl, HERE / "custom_task.bddl")
        worker.env.set_init_state(original)
        worker.env.sim.forward()
        source_model = worker.env.sim.model
        raw_poses = object_poses(worker.env)
        custom_env = worker._env_class(bddl_file_name=str(HERE / "custom_task.bddl"),
            camera_heights=cfg.render_size, camera_widths=cfg.render_size,
            control_freq=20, horizon=1000, ignore_done=True, controller="OSC_POSE", robots=["Panda"])
        custom_env.seed(7)
        custom_env.reset()
        model = custom_env.sim.model
        sampled_fixture = fixture_snapshot(custom_env)
        if not np.allclose(sampled_fixture["body_xyz_m"], original_fixture["body_xyz_m"], atol=1e-12, rtol=0):
            raise RuntimeError("Appending obstacle changed fixed stove placement")
        custom = np.asarray(custom_env.sim.get_state().flatten()).copy()
        custom[0] = original[0]
        mapped = []
        for i in range(source_model.njnt):
            name = source_model.joint_id2name(i)
            source_q, target_q = address(source_model, name), address(model, name)
            source_v, target_v = address(source_model, name, True), address(model, name, True)
            assert source_q[1] - source_q[0] == target_q[1] - target_q[0]
            assert source_v[1] - source_v[0] == target_v[1] - target_v[0]
            custom[1+target_q[0]:1+target_q[1]] = original[1+source_q[0]:1+source_q[1]]
            custom[1+model.nq+target_v[0]:1+model.nq+target_v[1]] = original[1+source_model.nq+source_v[0]:1+source_model.nq+source_v[1]]
            mapped.append({"joint": name, "source_qpos_address": list(source_q),
                           "target_qpos_address": list(target_q), "source_qvel_address": list(source_v),
                           "target_qvel_address": list(target_v)})
        assert source_model.na == model.na
        if model.na:
            custom[-model.na:] = original[-source_model.na:]
        can_joint = custom_env.env.objects_dict[CAN].joints[0]
        cq, cv = address(model, can_joint), address(model, can_joint, True)
        quat = np.array([2**-.5, 2**-.5, 0., 0.])
        custom[1+cq[0]:1+cq[1]] = np.r_[0., 0., 1., quat]
        custom[1+model.nq+cv[0]:1+model.nq+cv[1]] = 0.
        custom_env.set_init_state(custom)
        custom_env.sim.forward()
        bounds = geom_bounds(custom_env, CAN)
        mug = np.array(original_settled["moka_pot_2"]["qpos"][:2])
        plate = np.array(original_fixture["cook_region_xyz_m"][:2])
        support_surface = table_surface(custom_env, (mug + plate) / 2)
        table_z = support_surface["upper_z_m"]
        bottom_relative_z = bounds["lower_xyz_m"][2] - 1.
        can_z = table_z - bottom_relative_z + .0002
        delta = plate - mug
        perpendicular = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        candidates = []
        selected = None
        for offset in (0., .01, -.01, .02, -.02, .03, -.03, .04, -.04):
            for fraction in (.45, .50, .40, .55, .35):
                xy = mug + fraction * delta + offset * perpendicular
                custom[1+cq[0]:1+cq[0]+3] = np.r_[xy, can_z]
                bad = []
                # Official raw states start objects above the table. Check the
                # original settling corridor, not only its elevated initial pose.
                for alpha in (0., .25, .5, .75, 1.):
                    probe = custom.copy()
                    for mapping in mapped:
                        sq, tq = mapping["source_qpos_address"], mapping["target_qpos_address"]
                        probe[1+tq[0]:1+tq[1]] = ((1-alpha)*original[1+sq[0]:1+sq[1]]
                            + alpha*settled_original_state[1+sq[0]:1+sq[1]])
                    custom_env.sim.set_state_from_flattened(probe)
                    custom_env.sim.forward()
                    bad = [c for c in contacts(custom_env) if CAN in (c["a"], c["b"])
                           and next(x for x in (c["a"], c["b"]) if x != CAN) not in ("world", "kitchen_table")]
                    if bad:
                        break
                candidates.append({"fraction_source_pot_to_cook_region": fraction, "perpendicular_offset_m": offset,
                    "xyz_m": [*xy.tolist(), can_z], "can_other_contact_count": len(bad),
                    "can_other_contacts": bad[:3], "first_blocked_settling_fraction": alpha if bad else None})
                if not bad:
                    selected = candidates[-1]
                    break
            if selected is not None:
                break
        write("placement_candidates.json", {"candidates": candidates, "selected": selected})
        if selected is None:
            raise RuntimeError("No tested near-centerline position fits without new contacts; original objects were not moved")
        custom_env.set_init_state(custom)
        custom_env.sim.forward()
        np.save(HERE / "initial_state.npy", custom, allow_pickle=False)
        for mapping in mapped:
            sq, tq = mapping["source_qpos_address"], mapping["target_qpos_address"]
            sv, tv = mapping["source_qvel_address"], mapping["target_qvel_address"]
            assert np.array_equal(original[1+sq[0]:1+sq[1]], custom[1+tq[0]:1+tq[1]])
            assert np.array_equal(original[1+source_model.nq+sv[0]:1+source_model.nq+sv[1]],
                                  custom[1+model.nq+tv[0]:1+model.nq+tv[1]])
        write("layout_manifest.json", {
            "kind": "custom_scene_evaluator_authoring_only", "task_key": "libero_10_t8",
            "prompt": task.language, "seed": 7, "episode_index": 0,
            "model_calls": 0, "policy_actions": 0, "original_objects_moved": False,
            "original_joint_qpos_qvel_bitwise_preserved": True, "mapped_joints": mapped,
            "original_state_length": len(original), "custom_state_length": len(custom),
            "official_state_file": str(source), "official_state_file_sha256": sha(source),
            "original_bddl_sha256": sha(HERE / "original_task.bddl"),
            "custom_bddl_sha256": sha(HERE / "custom_task.bddl"),
            "original_initial_state_sha256": sha(HERE / "original_initial_state.npy"),
            "custom_initial_state_sha256": sha(HERE / "initial_state.npy"),
            "task_contract": contract, "original_raw_object_poses": raw_poses,
            "original_fixture": original_fixture, "custom_raw_fixture": fixture_snapshot(custom_env),
            "original_settled_object_poses": original_settled,
            "custom_raw_object_poses": object_poses(custom_env),
            "bottle": {"name": CAN, "asset": "ketchup", "orientation_wxyz": quat.tolist(),
                    "collision_bounds": geom_bounds(custom_env, CAN), "table_height_m": table_z,
                    "table_support_geometry": support_surface,
                    "workspace_reference_height_m": float(custom_env.env.workspace_offset[2]),
                    "placement": selected, "source_to_destination_centerline_xy_m": [mug.tolist(), plate.tolist()],
                    "source_object": "moka_pot_2", "destination_site": "flat_stove_1_cook_region",
                    "bootstrap_region_note": "BDDL constructor seed only; saved initial_state.npy defines final placement"},
            "simulator_configuration": asdict(config(True)[0]),
            "authoring_note": "Bottle geometry is evaluator-only scene construction; no geometry is supplied to any policy. No task rollout occurred.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({"built": str(HERE), "can_placement": selected}), flush=True)
    finally:
        if custom_env is not None:
            custom_env.close()
        worker.close()


def fixture_snapshot(env):
    model, data = env.sim.model, env.sim.data
    name = "flat_stove_1"
    obj = env.env.fixtures_dict[name]
    bid = model.body_name2id(obj.root_body)
    sid = model.site_name2id("flat_stove_1_cook_region")
    value = float(data.get_joint_qpos("flat_stove_1_button"))
    return {"name": name, "body_xyz_m": data.body_xpos[bid].tolist(),
            "body_quat_wxyz": data.body_xquat[bid].tolist(),
            "model_body_pos": model.body_pos[bid].tolist(), "model_body_quat": model.body_quat[bid].tolist(),
            "button_qpos": value, "is_on": value >= .5,
            "cook_region_xyz_m": data.site_xpos[sid].tolist(),
            "cook_region_halfsize_m": model.site_size[sid].tolist()}


def external_highres(env, prefix):
    import numpy as np
    from PIL import Image
    frame = env.sim.render(camera_name="agentview", width=1024, height=1024)
    Image.fromarray(np.ascontiguousarray(frame[::-1])).save(HERE / f"{prefix}external_1024.png")


def top_down(env, prefix="custom_"):
    import numpy as np
    from PIL import Image
    model = env.sim.model
    cid = model.camera_name2id("agentview")
    old_pos, old_quat, old_fovy = model.cam_pos[cid].copy(), model.cam_quat[cid].copy(), float(model.cam_fovy[cid])
    try:
        model.cam_pos[cid] = [0., 0., 2.2]
        model.cam_quat[cid] = [1., 0., 0., 0.]
        model.cam_fovy[cid] = 48.
        env.sim.forward()
        frame = env.sim.render(camera_name="agentview", width=1024, height=1024)
        Image.fromarray(np.ascontiguousarray(frame[::-1])).save(HERE / f"{prefix}top_down.png")
    finally:
        model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = old_pos, old_quat, old_fovy
        env.sim.forward()


def preview():
    import numpy as np
    from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
    from gpt_field.agent_speedup_v2_user_check_sim import UserCheckLiberoWorker
    cfg, monitor = config(True)
    worker = UserCheckLiberoWorker(cfg, monitor)
    try:
        response = worker.reset(8, 0)
        write("initial_observation.json", response["observation"])
        digest = initial_observation_sha256(response["observation"])
        (HERE / "initial_observation.sha256").write_text(digest + "\n")
        images(response["observation"], "custom_")
        external_highres(worker.env, "custom_")
        env, model, data = worker.env, worker.env.sim.model, worker.env.sim.data
        initial_poses = object_poses(env)
        initial_fixture = fixture_snapshot(env)
        initial_contacts = contacts(env)
        can_initial = np.array(initial_poses[CAN]["qpos"])
        # A short, stationary open-gripper hold checks passive layout stability;
        # it is evaluator setup, never a task-policy action or task test.
        all_contacts = list(initial_contacts)
        for _ in range(10):
            worker.step([0., 0., 0., 0., 0., 0., -1.])
            all_contacts.extend(contacts(env))
        final_poses = object_poses(env)
        can_final = np.array(final_poses[CAN]["qpos"])
        bid = model.body_name2id(env.env.objects_dict[CAN].root_body)
        upright = float(data.body_xmat[bid].reshape(3, 3)[2, 1])
        can_contacts = [c for c in contacts(env) if CAN in (c["a"], c["b"])]
        support = []
        for c in can_contacts:
            other = c["b"] if c["a"] == CAN else c["a"]
            upward = -c["normal"][2] if c["a"] == CAN else c["normal"][2]
            if other in ("world", "kitchen_table") and upward > .95:
                support.append(c)
        bad = [c for c in all_contacts if CAN in (c["a"], c["b"])
               and next(x for x in (c["a"], c["b"]) if x != CAN) not in ("world", "kitchen_table")]
        robot_object = [c for c in all_contacts if "robot" in (c["a"], c["b"])
                        and (c["a"] in env.env.objects_dict or c["b"] in env.env.objects_dict)]
        drift = float(np.linalg.norm(can_final[:3] - can_initial[:3]))
        angular_drift = float(2*np.arccos(np.clip(abs(np.dot(can_final[3:], can_initial[3:])), 0., 1.)))
        manifest = json.loads((HERE / "layout_manifest.json").read_text())
        preserved = {}
        for name, baseline in manifest["original_settled_object_poses"].items():
            before = np.asarray(baseline["qpos"])
            after = np.asarray(initial_poses[name]["qpos"])
            preserved[name] = {"translation_delta_m": float(np.linalg.norm(after[:3]-before[:3])),
                "orientation_delta_rad": float(2*np.arccos(np.clip(abs(np.dot(after[3:],before[3:])),0.,1.)))}
        original_fixture = manifest["original_fixture"]
        fixture_difference = {"translation_delta_m": float(np.linalg.norm(np.asarray(initial_fixture["body_xyz_m"])-original_fixture["body_xyz_m"])),
            "orientation_delta_rad": float(2*np.arccos(np.clip(abs(np.dot(initial_fixture["body_quat_wxyz"],original_fixture["body_quat_wxyz"])),0.,1.))),
            "button_qpos_delta": abs(initial_fixture["button_qpos"]-original_fixture["button_qpos"]),
            "original_on": original_fixture["is_on"], "custom_on": initial_fixture["is_on"]}
        checks = {"stove_fixture_pose_preserved": fixture_difference["translation_delta_m"] < 1e-12 and fixture_difference["orientation_delta_rad"] < 1e-7,
                  "stove_on_state_preserved": initial_fixture["is_on"] == original_fixture["is_on"] and fixture_difference["button_qpos_delta"] < 1e-12,
                  "initial_goal_false": not response["status"]["libero_success"],
                  "no_new_can_object_or_robot_contact": not bad,
                  "no_robot_object_contact": not robot_object,
                  "can_table_supported": bool(support), "can_upright_cosine_over_0_995": upright > .995,
                  "can_passive_drift_under_2mm": drift < .002,
                  "can_passive_rotation_under_1deg": bool(angular_drift < np.deg2rad(1.)),
                  "original_objects_preserved_after_reset": all(x["translation_delta_m"] < .003
                      and x["orientation_delta_rad"] < np.deg2rad(2.) for x in preserved.values())}
        manifest.update({"initial_observation_sha256": digest, "initial_status": response["status"],
            "standard_reset_settle_actions": cfg.settle_steps, "additional_passive_hold_actions": 10,
            "custom_initial_settled_object_poses": initial_poses,
            "custom_initial_fixture": initial_fixture, "fixture_comparison": fixture_difference,
            "custom_after_passive_hold_object_poses": final_poses,
            "original_object_settled_pose_comparison": preserved,
            "checks": checks, "checks_passed": all(checks.values()),
            "can_passive_translation_drift_m": drift, "can_passive_rotation_drift_rad": angular_drift,
            "can_local_y_world_z_cosine": upright, "can_table_support_contacts": support,
            "new_can_contacts": bad, "robot_object_contacts": robot_object,
            "can_after_hold_bounds": geom_bounds(env, CAN),
            "preview": {"external": "custom_external.png", "external_1024": "custom_external_1024.png", "wrist": "custom_wrist.png",
                        "top_down": "custom_top_down.png", "top_down_evaluator_only": True},
        })
        write("layout_manifest.json", manifest)
        # Save the top-down authoring view only after exact policy RGB/hash.
        top_down(env)
        print(json.dumps({"preview": str(HERE), "checks": checks,
                          "initial_observation_sha256": digest}), flush=True)
    finally:
        worker.close()


def probe():
    import numpy as np
    import torch
    from gpt_field.agent_speedup_v2_user_check_sim import UserCheckLiberoWorker
    cfg, monitor = config()
    worker = UserCheckLiberoWorker(cfg, monitor)
    try:
        response = worker.reset(8, 0)
        images(response["observation"], "original_")
        write("original_observation.json", response["observation"])
        top_down(worker.env, "original_")
        task = worker.benchmark.get_task(8)
        source = worker.benchmark_root / "init_files" / task.problem_folder / task.init_states_file
        original = np.asarray(torch.load(source, map_location="cpu", weights_only=False)[0]).copy()
        np.save(HERE / "original_initial_state.npy", original, allow_pickle=False)
        original_bddl = Path(worker.benchmark.get_task_bddl_file_path(8))
        (HERE / "original_task.bddl").write_bytes(original_bddl.read_bytes())
        m,d=worker.env.sim.model,worker.env.sim.data
        objects={}
        for name,obj in {**worker.env.env.objects_dict,**worker.env.env.fixtures_dict}.items():
            bid=m.body_name2id(obj.root_body)
            objects[name]={"body_xyz_m":d.body_xpos[bid].tolist(),"body_quat_wxyz":d.body_xquat[bid].tolist(),"root_body":obj.root_body,"joints":obj.joints}
        write("original_geometry.json",{"objects":objects,"object_poses":object_poses(worker.env),"table_surface":table_surface(worker.env,[0,0]),"contacts":contacts(worker.env),"status":response["status"]})
        print(json.dumps(objects),flush=True)
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "build", "preview"))
    args = parser.parse_args()
    if args.phase:
        return {"probe": probe, "build": build, "preview": preview}[args.phase]()
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="3", PYOPENGL_PLATFORM="egl")
    for phase in ("build", "preview"):
        with (HERE / f"{phase}.log").open("w") as log:
            subprocess.run([SIM_PYTHON, "-u", str(Path(__file__).resolve()), "--phase", phase],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {phase}; see {HERE / (phase + '.log')}", flush=True)


if __name__ == "__main__":
    main()
