"""Native simulator smoke of the default ordered loader (128 px for speed).

Two independent context-manager loads, one passive hold action, no policy/model
calls. This validates wrapper integration, not a robot completion trajectory.
The separate scene builder checks repeated native 512 px policy observations.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from load_task import HERE, HOLD, PROMPT, TASK_KEY, load_task
from ordered_success import OrderedTaskEnv

sys.path.insert(0, str(HERE.parents[2] / "src"))
from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
from gpt_field.observation import PolicyObservation


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_names = ("custom_task.bddl", "initial_state.npy", "load_task.py", "ordered_success.py")
    hashes = {name: file_sha(HERE / name) for name in source_names}
    config_before = os.environ.get("LIBERO_CONFIG_PATH")
    records = []
    states = []
    checks = {}
    for i in range(2):
        with load_task(render_size=128) as (env, obs, prompt):
            state = np.asarray(env.sim.get_state().flatten()).copy()
            states.append(state)
            predicates = env.goal_predicates()
            progress = env.ordered_progress()
            record = {
                "context_index": i,
                "prompt": prompt,
                "external_camera_fovy_deg": float(env.sim.model.cam_fovy[
                    env.sim.model.camera_name2id("agentview")]),
                "observation_sha256": initial_observation_sha256(
                    PolicyObservation.from_libero(obs, prompt, image_size=128).to_dict()),
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "initial_predicates": predicates,
                "initial_progress": progress,
            }
            checks[f"context_{i}_ordered_wrapper"] = isinstance(env, OrderedTaskEnv)
            checks[f"context_{i}_correct_prompt"] = prompt == PROMPT
            checks[f"context_{i}_valid_initial_state"] = progress["initial_state_valid"]
            checks[f"context_{i}_no_initial_progress_or_success"] = (
                progress["completed_stage_count"] == 0
                and not env.check_success() and not env.raw_goal_success())
            checks[f"context_{i}_initial_predicates"] = (
                predicates["drawer_closed"] and predicates["bowl_inside_drawer"]
                and predicates["stove_off"] and not predicates["stove_on"]
                and not predicates["drawer_open"] and not predicates["bowl_on_cabinet_top"]
                and not predicates["moka_on_stove"])
            if i == 0:
                result = env.step(HOLD)
                after = env.ordered_progress()
                record["after_passive_hold"] = after
                checks["passive_step_returns_four_items"] = len(result) == 4
                checks["passive_step_preserves_initial_predicates"] = env.goal_predicates() == predicates
                checks["passive_step_observed_without_progress"] = (
                    after["observation_index"] == 1 and after["completed_stage_count"] == 0
                    and after["initial_state_valid"] and not after["ordered_success"])
            else:
                env.set_init_state(state)
                invalidated = env.ordered_progress()
                record["after_state_replacement"] = invalidated
                checks["native_state_replacement_invalidates_progress"] = (
                    not invalidated["initial_state_valid"]
                    and invalidated["invalid_reason"] == "set_init_state_requires_reset_progress"
                    and not env.check_success())
                reset = env.reset_progress()
                record["after_reset_progress"] = reset
                checks["native_reset_progress_reestablishes_baseline"] = (
                    reset["initial_state_valid"] and reset["completed_stage_count"] == 0
                    and reset["observation_index"] == 0 and not env.check_success())
            records.append(record)
        checks[f"context_{i}_config_environment_restored"] = os.environ.get("LIBERO_CONFIG_PATH") == config_before
    checks["independent_context_states_bitwise_equal"] = bool(np.array_equal(*states))
    checks["independent_context_observations_equal"] = records[0]["observation_sha256"] == records[1]["observation_sha256"]
    checks["frozen_inputs_unchanged"] = hashes == {name: file_sha(HERE / name) for name in source_names}
    output = {
        "kind": "native_ordered_loader_smoke",
        "task_key": TASK_KEY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "render_size": 128,
        "python_executable": sys.executable,
        "policy_actions": 0,
        "model_calls": 0,
        "passive_hold_actions_after_reset": 1,
        "loader_settle_actions_per_context": 10,
        "source_sha256": hashes,
        "contexts": records,
        "checks": checks,
        "all_passed": all(checks.values()),
        "scope": "Default ordered wrapper integration at 128 px; two resets and one passive step. No policy completion attempt.",
    }
    (HERE / "loader_validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_passed": output["all_passed"], "checks": checks}))
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
