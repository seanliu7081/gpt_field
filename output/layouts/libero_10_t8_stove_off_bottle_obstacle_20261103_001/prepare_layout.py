"""Derive and validate the stove-off task from the frozen t08 obstacle layout.

Scene construction, passive settling, and evaluator-only goal counterfactuals;
no model calls or policy rollout. Run with the simulator Python (README.md).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from load_task import HERE, HOLD, LIBERO_ROOT, PROMPT, SEED, SETTLE_STEPS, TASK_KEY, load_task

SOURCE = HERE.parent / "libero_10_t8_bottle_obstacle_20261103_001"
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
from gpt_field.libero import _load_task_bddl
from gpt_field.observation import PolicyObservation

BUTTON = "flat_stove_1_button"
OFF_QPOS = -0.0025  # Strict Turnoff predicate is qpos < 0; joint range starts at -0.005.
CAN = "ketchup_obstacle_1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def policy_observation(raw, prompt):
    return PolicyObservation.from_libero(raw, prompt, image_size=512).to_dict()


def load_helpers():
    # Reuse the frozen scene's evaluator/render helpers without running its main
    # or writing into its directory. Its exact source hash is in our manifest.
    spec = importlib.util.spec_from_file_location("source_t08_scene", SOURCE / "prepare_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module


def pose_comparison(before, after):
    result = {}
    for name, old in before.items():
        p, q = np.asarray(old["qpos"]), np.asarray(after[name]["qpos"])
        result[name] = {"qpos_bitwise_equal": bool(np.array_equal(p, q)),
                        "translation_delta_m": float(np.linalg.norm(p[:3] - q[:3])),
                        "quaternion_max_abs_delta": float(np.max(np.abs(p[3:] - q[3:])))}
    return result


def main():
    helper = load_helpers()
    source_hashes = {p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()}
    source_manifest = json.loads((SOURCE / "layout_manifest.json").read_text())
    original_text = (SOURCE / "custom_task.bddl").read_text()
    old_language = "(:language put both moka pots on the stove)"
    old_init = "    (Turnon flat_stove_1)\n"
    assert original_text.count(old_language) == original_text.count(old_init) == 1
    changed_text = original_text.replace(old_language, f"(:language {PROMPT})", 1).replace(
        old_init, "    (Turnoff flat_stove_1)\n", 1)
    (HERE / "custom_task.bddl").write_text(changed_text)
    source_sections, _ = _load_task_bddl(SOURCE / "custom_task.bddl")
    new_sections, _ = _load_task_bddl(HERE / "custom_task.bddl")
    assert all(new_sections[k] == v for k, v in source_sections.items()
               if k not in (":language", ":init"))
    expected_init = [["turnoff", "flat_stove_1"] if item == ["turnon", "flat_stove_1"] else item
                     for item in source_sections[":init"]]
    assert new_sections[":init"] == expected_init
    original_state = np.load(SOURCE / "initial_state.npy", allow_pickle=False)

    with load_task(bddl_path=SOURCE / "custom_task.bddl",
                   state_path=SOURCE / "initial_state.npy") as (env, raw, prompt):
        source_obs = policy_observation(raw, prompt)
        source_digest = initial_observation_sha256(source_obs)
        assert source_digest == source_manifest["initial_observation_sha256"]
        model = env.sim.model
        qi, vi = int(model.get_joint_qpos_addr(BUTTON)), int(model.get_joint_qvel_addr(BUTTON))
        nq, nv, na = int(model.nq), int(model.nv), int(model.na)
        baseline_poses = helper.object_poses(env)
        baseline_fixture = helper.fixture_snapshot(env)
        baseline_settled = np.asarray(env.sim.get_state().flatten()).copy()
        helper.external_highres(env, "source_")
    custom_state = original_state.copy()
    custom_state[1 + qi] = OFF_QPOS
    custom_state[1 + nq + vi] = 0.
    allowed_indices = [1 + qi, 1 + nq + vi]
    keep = np.ones(original_state.size, dtype=bool)
    keep[allowed_indices] = False
    assert np.array_equal(original_state[keep], custom_state[keep])
    np.save(HERE / "initial_state.npy", custom_state, allow_pickle=False)

    with load_task() as (env, raw, prompt):
        assert prompt == PROMPT
        model = env.sim.model
        assert (model.nq, model.nv, model.na) == (nq, nv, na)
        assert (model.get_joint_qpos_addr(BUTTON), model.get_joint_qvel_addr(BUTTON)) == (qi, vi)
        obs = policy_observation(raw, prompt)
        digest = initial_observation_sha256(obs)
        write("initial_observation.json", obs)
        (HERE / "initial_observation.sha256").write_text(digest + "\n")
        helper.images(obs, "custom_")
        helper.external_highres(env, "custom_")
        helper.top_down(env)
        initial_poses = helper.object_poses(env)
        initial_fixture = helper.fixture_snapshot(env)
        settled = np.asarray(env.sim.get_state().flatten()).copy()
        np.save(HERE / "settled_state.npy", settled, allow_pickle=False)
        initial_success = bool(env.check_success())
        initial_stove_off = bool(env.env.object_states_dict["flat_stove_1"].turn_off())
        initial_stove_on = bool(env.env.object_states_dict["flat_stove_1"].turn_on())
        all_contacts = helper.contacts(env)
        for _ in range(10):
            env.step(HOLD)
            all_contacts.extend(helper.contacts(env))
        final_poses = helper.object_poses(env)
        final_fixture = helper.fixture_snapshot(env)
        bottle_before = np.asarray(initial_poses[CAN]["qpos"])
        bottle_after = np.asarray(final_poses[CAN]["qpos"])
        drift = float(np.linalg.norm(bottle_after[:3] - bottle_before[:3]))
        angular_drift = float(2 * np.arccos(np.clip(abs(np.dot(bottle_after[3:], bottle_before[3:])), 0, 1)))
        bid = model.body_name2id(env.env.objects_dict[CAN].root_body)
        upright = float(env.sim.data.body_xmat[bid].reshape(3, 3)[2, 1])
        support = []
        for c in helper.contacts(env):
            if CAN not in (c["a"], c["b"]):
                continue
            other = c["b"] if c["a"] == CAN else c["a"]
            upward = -c["normal"][2] if c["a"] == CAN else c["normal"][2]
            if other in ("world", "kitchen_table") and upward > .95:
                support.append(c)
        bad = [c for c in all_contacts if CAN in (c["a"], c["b"])
               and next(x for x in (c["a"], c["b"]) if x != CAN) not in ("world", "kitchen_table")]
        comparison = pose_comparison(baseline_poses, initial_poses)
        fixture_keys = ("body_xyz_m", "body_quat_wxyz", "model_body_pos", "model_body_quat",
                        "cook_region_xyz_m", "cook_region_halfsize_m")
        checks = {
            "source_reset_reproduced": source_digest == source_manifest["initial_observation_sha256"],
            "only_stove_joint_changed_in_saved_state": bool(np.array_equal(original_state[keep], custom_state[keep])),
            "layout_and_goal_bddl_preserved": all(new_sections[k] == v for k, v in source_sections.items()
                                                   if k not in (":language", ":init")),
            "new_task_language_loaded": prompt == PROMPT,
            "stove_off_after_reset": initial_stove_off and not initial_stove_on,
            "stove_off_after_passive_hold": bool(env.env.object_states_dict["flat_stove_1"].turn_off()),
            "initial_goal_false": not initial_success,
            "all_object_poses_preserved_after_reset": all(v["qpos_bitwise_equal"] for v in comparison.values()),
            "stove_fixture_and_cook_region_preserved": all(initial_fixture[k] == baseline_fixture[k] for k in fixture_keys),
            "other_settled_state_values_preserved": bool(np.array_equal(baseline_settled[keep], settled[keep])),
            "no_bottle_object_or_robot_contact": not bad,
            "bottle_table_supported": bool(support),
            "bottle_upright": upright > .995,
            "bottle_passive_translation_under_2mm": drift < .002,
            "bottle_passive_rotation_under_1deg": angular_drift < float(np.deg2rad(1.)),
        }
        from validate_goals import validate_goals
        goal_validation = validate_goals(env)
        write("goal_validation.json", goal_validation)
        checks["goal_truth_table_passed"] = bool(goal_validation["all_passed"])

    # A fresh construction must reproduce the same full policy observation.
    with load_task() as (env, raw, prompt):
        repeated_digest = initial_observation_sha256(policy_observation(raw, prompt))
    checks["repeat_reset_observation_identical"] = repeated_digest == digest
    checks["all_source_artifacts_unchanged"] = all(sha(SOURCE / name) == value for name, value in source_hashes.items())
    manifest = {
        "kind": "custom_task_evaluator_authoring_only", "task_key": TASK_KEY,
        "prompt": PROMPT, "source_task_key": "libero_10_t8", "source_layout": str(SOURCE),
        "seed": SEED, "episode_index": 0, "standard_reset_settle_actions": SETTLE_STEPS,
        "additional_passive_hold_actions": 10, "model_calls": 0, "policy_actions": 0,
        "custom_bddl_sha256": sha(HERE / "custom_task.bddl"),
        "custom_initial_state_sha256": sha(HERE / "initial_state.npy"),
        "initial_observation_sha256": digest, "repeat_initial_observation_sha256": repeated_digest,
        "source_artifact_sha256": source_hashes,
        "task_contract": {"language": PROMPT, "init": new_sections[":init"], "goal": new_sections[":goal"]},
        "stove_change": {"joint": BUTTON, "qpos_address": qi, "qvel_address": vi,
                         "source_qpos": float(original_state[1 + qi]), "new_qpos": OFF_QPOS,
                         "new_qvel": 0., "changed_flat_state_indices": np.flatnonzero(original_state != custom_state).tolist(),
                         "off_predicate": "button qpos < 0", "on_predicate": "button qpos >= 0.5"},
        "model_dimensions": {"nq": nq, "nv": nv, "na": na},
        "source_initial_fixture": baseline_fixture, "custom_initial_fixture": initial_fixture,
        "custom_after_passive_hold_fixture": final_fixture,
        "custom_initial_settled_object_poses": initial_poses, "custom_after_passive_hold_object_poses": final_poses,
        "source_object_pose_comparison": comparison, "bottle": source_manifest["bottle"],
        "bottle_passive_translation_drift_m": drift, "bottle_passive_rotation_drift_rad": angular_drift,
        "bottle_table_support_contacts": support, "new_bottle_contacts": bad,
        "checks": checks, "checks_passed": all(checks.values()),
        "simulator_configuration": {"libero_root": str(LIBERO_ROOT), "python_executable": sys.executable,
                                    "controller": "OSC_POSE", "control_freq_hz": 20, "render_size": 512},
        "validation_scope": "Passive scene/reset checks and synthetic evaluator goal truth table; no policy completion test.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write("layout_manifest.json", manifest)
    if not all(checks.values()):
        raise RuntimeError(f"Layout checks failed: {[k for k, v in checks.items() if not v]}")
    print(json.dumps({"task_key": TASK_KEY, "checks_passed": len(checks), "initial_observation_sha256": digest}))


if __name__ == "__main__":
    main()
