"""Check official resets and camera geometry without credentials or model calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from . import run
from .libero10_tasks import LIBERO10_TASK_KEYS, RESET_SHA256
from .policy import policy_instructions
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.experiment import write_json


def check(key, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = resolve_distance_task(key)
    config = run.sim_configuration(key)
    report = {"task_key": key, "task_id": task.task_id, "suite": task.suite,
              "model_calls": 0, "policy_actions": 0, "credentials_loaded": False,
              "simulator": asdict(config), "passed": False}
    sim = None
    try:
        with run.life._simulator_without_credentials():
            sim = run.CalibratedSimProcess(config)
            sim.start()
        official = sim.list_tasks()
        assert official[task.task_id]["prompt"] == task.prompt
        obs = sim.reset(task.task_id, 0)
        actual = run.initial_identity(obs)
        assert obs.prompt == task.prompt
        assert actual == RESET_SHA256[(key, 0)], (key, actual)
        geometry = sim.geometry()
        request = run.model_request(obs, 0, geometry)
        assert set(request) == {"external_image", "wrist_image", "robot_state",
                               "joint_positions", "geometry", "step"}
        assert len(request["joint_positions"]) == 7
        assert sim.status["steps"] == 0 and not sim.status["terminated"]
        assert sim.status["collision"]["mode"] == "generic_contact_v2"
        assert sim.status["collision"]["collision_policy"] == "record_only"
        write_json(output / "initial_observation.json", obs.to_dict())
        write_json(output / "geometry.json", geometry)
        write_json(output / "simulator.json", sim.metadata)
        (output / "policy_prompt.txt").write_text(policy_instructions(key))
        report.update(passed=True, initial_sha256=actual,
                      frozen_reset_matched=True, official_prompt=obs.prompt,
                      collision_mode=sim.status["collision"]["mode"])
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if sim:
            sim.close()
        write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=LIBERO10_TASK_KEYS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.task, args.output), indent=2))
